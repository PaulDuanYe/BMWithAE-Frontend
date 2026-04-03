async function showParameterSelectionPanel() {
  const graphPanel = $('#graphSelectionPanel');
  const parameterPanel = $('#parameterSelectionPanel');

  if (!graphPanel) {
    console.error('graphPanel not found');
    return;
  }

  if (!parameterPanel) {
    console.error('parameterPanel not found');
    return;
  }

  try {
    graphPanel.classList.add('slide-out');
    parameterPanel.style.display = 'block';

    setTimeout(() => {
      parameterPanel.classList.add('slide-in');

    }, 50);
  } catch (error) {
    console.error('Failed to initialize parameter panel:', error);
    alert('Failed to load parameter data. Please try again.');

    graphPanel.classList.remove('slide-out');
    parameterPanel.classList.remove('slide-in');
    parameterPanel.style.display = 'none';
  }
}

function hideParameterSelectionPanel() {
    const graphPanel = $('#graphSelectionPanel');
    const parameterPanel = $('#parameterSelectionPanel');
    
    // Slide out explorer panel
    parameterPanel.classList.remove('slide-in');
    
    // Slide in load panel
    graphPanel.classList.remove('slide-out');
    
    // Hide explorer panel after animation
    setTimeout(() => {
    parameterPanel.style.display = 'none';
    }, 400);
}

function createSelectors(selectors) {
  const container = document.getElementById("parameterContent");

  if (!container) {
      console.error('Container "parameterContent" not found.');
      return;
  }

  container.innerHTML = "";

  selectors.forEach((selectorConfig, index) => {
    console.log(selectorConfig);
      const wrapper = document.createElement("div");
      wrapper.className = "explorer-section";

      const label = document.createElement("label");
      label.className = "input-label";

      const selectId = selectorConfig.id || `selector-${index}`;
      label.setAttribute("for", selectId);
      label.textContent = selectorConfig.name;

      const select = document.createElement("select");
      select.className = "selector-input";
      select.id = selectId;

      selectorConfig.options.forEach(optionValue => {
          const option = document.createElement("option");
          option.value = optionValue;
          option.textContent = optionValue;
          select.appendChild(option);
      });

      wrapper.appendChild(label);
      wrapper.appendChild(select);
      container.appendChild(wrapper);
  });
}

$('#btnBackToSelection').addEventListener('click', hideParameterSelectionPanel);