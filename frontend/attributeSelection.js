function getSelectedProtectedAttributes() {
  const checkboxes = document.querySelectorAll('.protected-attr-checkbox:checked');
  return Array.from(checkboxes).map(cb => cb.value);
}

function updateProtectedAttrDisplay() {
  const selected = getSelectedProtectedAttributes();
  const textElement = $('#protectedAttrText');
  
  if (selected.length === 0) {
    textElement.textContent = 'Select attributes...';
    textElement.classList.add('placeholder');
  } else {
    textElement.textContent = selected.join(', ');
    textElement.classList.remove('placeholder');
  }
}

function openAttrModal() {
  $('#attrModalOverlay').style.display = 'flex';
}

function closeAttrModal() {
  $('#attrModalOverlay').style.display = 'none';
}

function confirmAttrSelection() {
  updateProtectedAttrDisplay();
  closeAttrModal();
  handleProtectedAttributesChange();
}
    
async function handleProtectedAttributesChange() {
  const selectedAttrs = getSelectedProtectedAttributes();
  
  if (selectedAttrs.length > 0) {
    console.log('[INFO] Protected attributes changed to:', selectedAttrs);
    
    // Update config with selected protected attributes
    config.protectedAttributes = selectedAttrs;
    
    // Recalculate and display bias metrics for the selected attributes
    await displayBiasOverview(selectedAttrs);
  } else {
    // No attributes selected, reset display
    await displayBiasOverview(null);
  }
}
