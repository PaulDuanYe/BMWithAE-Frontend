"""
module_transform.py
=================
This module provides a Transform class for transforming datasets based on a specified changed_dict.
The class handles both numerical and categorical attributes, applying transformations
as defined in the changed_dict parameter. Also handles feature scaling when required by the model.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from core_config import PARAMS_MAIN_ALPHA_O, PARAMS_TRANSFORM_STREAM_CONFIG, PARAMS_TRANSFORM_STREAM, PARAMS_TRANSFORM, PARAMS_TRANSFORM_MULTI, PARAMS_MAIN_CLASSIFIER


class Transform:
    """
    A class for transforming data based on a changed_dict specification.
    
    This class transforms data according to the provided changed_dict,
    which defines transformations for numerical and categorical attributes.
    Also handles feature scaling for models that require it.
    """
    
    # Models that require feature scaling (gradient-based or distance-based)
    MODELS_REQUIRING_SCALING = {'LR', 'SVM', 'KNN', 'MLP', 'LDA', 'QDA'}
    
    def __init__(self):
        """
        Initialize a new Transform instance.
        
        Automatically generates stream data based on configurations from config.py.
        Initializes scaler if the current model requires it.
        """
        # Generate stream_data based on stream_name from config.py
        self.stream_data = self.gen_stream(PARAMS_TRANSFORM_STREAM, **PARAMS_TRANSFORM_STREAM_CONFIG)

        # Validate and set alpha_O parameter from config.py
        self.alpha_O = PARAMS_MAIN_ALPHA_O if 0 < PARAMS_MAIN_ALPHA_O < 1 else 0.8
        
        # Initialize scaler if current model requires it
        self.scaler = None
        self.scaler_fitted = False
        if self._model_requires_scaling():
            self.scaler = StandardScaler()
    
    @staticmethod
    def _model_requires_scaling():
        """
        Check if the current classifier requires feature scaling.
        
        Returns:
            bool: True if the model requires scaling, False otherwise
        """
        return PARAMS_MAIN_CLASSIFIER in Transform.MODELS_REQUIRING_SCALING


    def _cal_beta_value(self, df_temp_attr, beta_value):
        """
        Calculate the beta transformation for a single attribute using the specified method.
        
        Parameters:
            df_temp_attr: The attribute data to transform
            beta_value: The beta value to use for transformation
            
        Returns:
            Transformed attribute data
        """
        if PARAMS_TRANSFORM == 'poly':
            # Apply polynomial transformation: |x|^beta * sign(x)
            result = (df_temp_attr.abs() ** beta_value) * (df_temp_attr.apply(np.sign))
            return result.astype(float)  # Ensure float type for consistency
        elif PARAMS_TRANSFORM == 'log':
            result = np.abs(np.log(df_temp_attr.abs() + 1e-5) / np.log(beta_value)) * (df_temp_attr.apply(np.sign))
            return result.astype(float)
        elif PARAMS_TRANSFORM == 'arcsin':
            max_value = df_temp_attr.abs().max()
            min_value = df_temp_attr.abs().min()
            df_temp_attr_norm = 2 * (df_temp_attr - min_value) / (max_value - min_value) - 1
            result = (np.arcsin(df_temp_attr_norm.abs() / max_value) ** beta_value) * (df_temp_attr_norm.apply(np.sign))
            return result.astype(float)
        else:
            raise ValueError(f"Invalid transform method: {PARAMS_TRANSFORM}")


    def _calculate_beta_transformation(self, df_temp_attr, beta_index):
        """
        Calculate beta transformation for a single beta value based on current PARAMS_TRANSFORM_MULTI.
        
        Parameters:
            df_temp_attr: The attribute data to transform
            beta_index: The index in stream_data (required for t2 and t3 transforms)
            
        Returns:
            Transformed attribute data
        """
        if PARAMS_TRANSFORM_MULTI == 't1':
            beta_value = self.stream_data[beta_index]
            return self._cal_beta_value(df_temp_attr, beta_value)
        elif PARAMS_TRANSFORM_MULTI == 't2':
            if isinstance(beta_index, int) and 0 <= beta_index < len(self.stream_data):
                beta_value = np.prod(self.stream_data[:beta_index+1])
                return self._cal_beta_value(df_temp_attr, beta_value)
            else:
                raise ValueError(f"Invalid beta_index: {beta_index}")
        elif PARAMS_TRANSFORM_MULTI == 't3':
            X_sum = 0
            if isinstance(beta_index, int) and 0 <= beta_index < len(self.stream_data):
                for i in range(beta_index + 1):
                    beta_i = self.stream_data[i]
                    X_sum += self._cal_beta_value(df_temp_attr, beta_i)
                return X_sum
            else:
                raise ValueError(f"Invalid beta_index: {beta_index}")
        else:
            raise ValueError(f"Invalid transform method: {PARAMS_TRANSFORM_MULTI}")


    def _transform_numerical_attribute(self, df_temp_attr, change):
        """
        Transform a single numerical attribute based on the provided change parameters.
        
        Parameters:
            df_temp_attr: The numerical attribute data to transform
            change: Transformation parameters for the attribute
            
        Returns:
            Transformed attribute data
        """
        # Get beta_O and beta_Y from stream_data if available
        if 'beta_O' in change.keys():
            beta_O_index = change['beta_O']
            if beta_O_index < 0:
                X_O = 1
            else:
                X_O = self._calculate_beta_transformation(df_temp_attr, beta_O_index)
        
        if 'beta_Y' in change.keys():
            beta_Y_index = change['beta_Y']
            if beta_Y_index < 0:
                X_Y = 1
            else:
                X_Y = self._calculate_beta_transformation(df_temp_attr, beta_Y_index)
        
        if 'beta_O' in change.keys() and 'beta_Y' in change.keys():
            return self.alpha_O * X_O + (1 - self.alpha_O) * X_Y
        elif 'beta_O' in change.keys():
            return X_O
        elif 'beta_Y' in change.keys():
            return X_Y
        
        # If no valid transformation parameters are provided, return original
        return df_temp_attr


    def _transform_categorical_attribute(self, df_temp_attr, re_bin_dict):
        """
        Transform a single categorical attribute based on the provided re-binning dictionary.
        
        Parameters:
            df_temp_attr: The categorical attribute data to transform
            re_bin_dict: Dictionary defining the re-binning rules
            
        Returns:
            Transformed attribute data
        """
        # Create a copy to avoid modifying the original
        transformed_attr = df_temp_attr.copy()
        
        # Apply re-binning
        for old_value, new_value in re_bin_dict.items():
            transformed_attr[transformed_attr == old_value] = new_value
            
        return transformed_attr


    def transform_data(self, X, changed_dict, num_attrs, cate_attrs, fit_scaler=False):
        """
        Transform the dataset according to the specified changed_dict.
        Also applies feature scaling if the model requires it.
        
        Parameters:
            X: DataFrame to be transformed
            changed_dict: Dictionary containing attributes to change and their transformation parameters
                          Format: {attribute_name: transformation_parameters}
            num_attrs: List of numerical attribute names
            cate_attrs: List of categorical attribute names
            fit_scaler: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            pandas.DataFrame: Transformed dataset
        """
        import pandas as pd
        
        # Create a copy of the data to avoid modifying the original
        df_data_temp = X.copy()

        # Step 1: Apply attribute-specific transformations from changed_dict
        for attribute in changed_dict.keys():
            # Skip scaler params (handled separately)
            if attribute == '__scaler__':
                continue
                
            df_temp_attr = df_data_temp[attribute]
            change = changed_dict[attribute]

            if attribute in num_attrs:
                # Process numerical attributes using the dedicated method
                
                if change == 'dropped':
                    df_temp_attr = 1
                else:
                    df_temp_attr = self._transform_numerical_attribute(df_temp_attr, change)
            elif attribute in cate_attrs:
                # Process categorical attributes using the dedicated method
                if change == 'dropped':
                    df_temp_attr = 1
                else:
                    re_bin_dict = changed_dict[attribute]
                    df_temp_attr = self._transform_categorical_attribute(df_temp_attr, re_bin_dict)

            # Update the attribute in the temporary DataFrame
            df_data_temp[attribute] = df_temp_attr

        # Step 2: Apply feature scaling if required by the model
        if self.scaler is not None:
            if fit_scaler:
                # Fit and transform (for training data)
                df_data_temp = pd.DataFrame(
                    self.scaler.fit_transform(df_data_temp),
                    columns=df_data_temp.columns,
                    index=df_data_temp.index
                )
                self.scaler_fitted = True
                
                # Store scaler parameters in changed_dict for reproducibility
                if '__scaler__' not in changed_dict:
                    changed_dict['__scaler__'] = {}
                changed_dict['__scaler__']['mean_'] = self.scaler.mean_.tolist()
                changed_dict['__scaler__']['scale_'] = self.scaler.scale_.tolist()
                changed_dict['__scaler__']['var_'] = self.scaler.var_.tolist()
                changed_dict['__scaler__']['n_features_in_'] = int(self.scaler.n_features_in_)
                changed_dict['__scaler__']['feature_names'] = df_data_temp.columns.tolist()
                
            elif self.scaler_fitted:
                # Only transform (for test/new data)
                df_data_temp = pd.DataFrame(
                    self.scaler.transform(df_data_temp),
                    columns=df_data_temp.columns,
                    index=df_data_temp.index
                )
            else:
                # Scaler exists but not fitted yet - try to load from changed_dict
                if '__scaler__' in changed_dict:
                    self._load_scaler_from_dict(changed_dict['__scaler__'])
                    df_data_temp = pd.DataFrame(
                        self.scaler.transform(df_data_temp),
                        columns=df_data_temp.columns,
                        index=df_data_temp.index
                    )
                # else: First time, will be fitted on next call with fit_scaler=True

        return df_data_temp
    
    def _load_scaler_from_dict(self, scaler_params):
        """
        Load scaler parameters from a dictionary.
        
        Parameters:
            scaler_params: Dictionary containing scaler parameters
        """
        import numpy as np
        if self.scaler is not None and scaler_params:
            self.scaler.mean_ = np.array(scaler_params['mean_'])
            self.scaler.scale_ = np.array(scaler_params['scale_'])
            self.scaler.var_ = np.array(scaler_params['var_'])
            self.scaler.n_features_in_ = scaler_params['n_features_in_']
            self.scaler_fitted = True


    def check_transform_validity(self, X, attribute, change, num_attrs, cate_attrs):
        """
        Check if transforming the specified attribute with the given change is valid.
        
        Parameters:
            X: Original DataFrame
            attribute: Attribute name to check
            change: Transformation parameters for the attribute
            num_attrs: List of numerical attribute names
            cate_attrs: List of categorical attribute names
            
        Returns:
            bool: True if transformation is valid, False otherwise
        """
        try:
            # Create a copy of the attribute to avoid modifying the original
            df_temp_attr = X[attribute].copy()
            
            # Apply transformation directly to the single attribute
            if attribute in num_attrs:
                df_transformed_attr = self._transform_numerical_attribute(df_temp_attr, change)
                
                # Check for infinity values
                if df_transformed_attr.isin([np.inf, -np.inf]).any():
                    print(f"[DEBUG] check_transform_validity FAILED for {attribute}: Contains infinity values")
                    return False
                      
                # Check for valid data types
                if not np.issubdtype(df_transformed_attr.dtype, np.number):
                    print(f"[DEBUG] check_transform_validity FAILED for {attribute}: Invalid data type {df_transformed_attr.dtype}")
                    return False
            
            elif attribute in cate_attrs:
                df_transformed_attr = self._transform_categorical_attribute(df_temp_attr, change)
                
                # Check for single unique value
                if df_transformed_attr.nunique() == 1:
                    print(f"[DEBUG] check_transform_validity FAILED for {attribute}: Only single unique value after transformation")
                    return False
                
        except Exception as e:
            print(f"[DEBUG] check_transform_validity FAILED for {attribute}: Exception {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        print(f"[DEBUG] check_transform_validity PASSED for {attribute} with change {change}")
        return True


    def gen_stream(self, mode, p=102, q=173, emin=1/2, emax=2, order=0, length=10):
        """
        Generate stream data based on the specified mode.
        
        Parameters:
            mode: Stream generation mode (d4A, d4B, E1-E9)
            p: Prime number for E modes (default: 102)
            q: Prime number for E modes (default: 173)
            emin: Minimum value for E modes (default: 1/2)
            emax: Maximum value for E modes (default: 2)
            order: Order of the stream (-1: descending, 0: original, 1: ascending)
            length: Length of the stream (default: 10, only for d4A and d4B)
            
        Returns:
            List: Generated stream data
        """
        if mode == 'd4A':
            # Generate 3, 5, 7, 9, ...
            return list(range(3, 3 + 2 * length, 2))
        elif mode == 'd4B':
            # Generate 1/3, 1/5, 1/7, 1/9, ...
            return [1 / x for x in range(3, 3 + 2 * length, 2)]
        elif mode.startswith('E') and mode[1:].isdigit() and 1 <= int(mode[1:]) <= 9:
            # Handle E1-E9 modes
            mode_num = int(mode[1:])
            stream_0 = []
            for num in range(q + 1):
                if (num != 0) and (num != 1):
                    temp_num = (p ** num) % q
                    if (temp_num != 1) and (temp_num != (q - 1)):
                        stream_0.append(temp_num)
            
            stream = []
            if mode_num == 1:
                for i in range(len(stream_0)):
                    temp_num_s = stream_0[i]
                    if emin < temp_num_s < emax:
                        stream.append(temp_num_s)
            elif mode_num == 2:
                for i in range(len(stream_0) - 1):
                    temp_num_s = stream_0[i] / stream_0[i + 1]
                    if emin < temp_num_s < emax:
                        stream.append(temp_num_s)
            elif mode_num == 3:
                for i in range(len(stream_0) - 1):
                    temp_num_s = stream_0[i] / (stream_0[i] + stream_0[i + 1])
                    if emin < temp_num_s < emax:
                        stream.append(temp_num_s)
            elif mode_num == 4:
                for i in range(len(stream_0) - 1):
                    temp_num_s = abs(stream_0[i] - stream_0[i + 1])
                    if emin < temp_num_s < emax:
                        stream.append(temp_num_s)
            elif mode_num == 5:
                for i in range(len(stream_0) - 1):
                    temp_num_s = np.abs((stream_0[i] - stream_0[i + 1]) / (stream_0[i] + stream_0[i + 1]))
                    if emin < temp_num_s < emax:
                        stream.append(temp_num_s)
            elif mode_num == 6:
                for i in range(len(stream_0)):
                    temp_num_s = 1 / stream_0[i]
                    if emin < temp_num_s < emax:
                        stream.append(temp_num_s)
            elif mode_num == 7:
                for i in range(len(stream_0) - 1):
                    temp_num_s = 1 / stream_0[i] + 1 / stream_0[i + 1]
                    if emin < temp_num_s < emax:
                        stream.append(temp_num_s)
            elif mode_num == 8:
                for i in range(len(stream_0) - 1):
                    temp_num_s = stream_0[i] * stream_0[i + 1] / (stream_0[i] + stream_0[i + 1])
                    if emin < temp_num_s < emax:
                        stream.append(temp_num_s)
            elif mode_num == 9:
                for i in range(len(stream_0) - 2):
                    temp_num_s = (stream_0[i] + stream_0[i + 1] + stream_0[i + 2]) / (stream_0[i] * stream_0[i + 1] * stream_0[i + 2])
                    if emin < temp_num_s < emax:
                        stream.append(temp_num_s)
            
            # Order the stream
            if order == -1:
                return list(np.sort(stream)[::-1])
            elif order == 0:
                return stream
            elif order == 1:
                return list(np.sort(stream))

        else:
            raise ValueError(f"Unsupported stream mode: {mode}")

