function getSelectedProtectedAttributes() {
  const checkboxes = document.querySelectorAll('.protected-attr-checkbox:checked');
  return Array.from(checkboxes).map(cb => cb.value);
}

function getSelectedTargetAttributes() {
  const el = document.querySelector('.target-attr-item.selected');
  return el ? el.dataset.attribute : null;
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

function updateTargetAttrDisplay() {
  const selected = getSelectedTargetAttributes();
  const textElement = $('#targetAttrText');
  
  if (!selected) {
    textElement.textContent = 'Select attributes...';
    textElement.classList.add('placeholder');
  } else {
    textElement.textContent = selected;
    textElement.classList.remove('placeholder');
  }
}

function openAttrModal() {
  $('#attrModalOverlay').style.display = 'flex';
}

function openTargetModal() {
  $('#targetModalOverlay').style.display = 'flex';
}

function closeAttrModal() {
  $('#attrModalOverlay').style.display = 'none';
}

function closeTargetModal() {
  $('#targetModalOverlay').style.display = 'none';
}

function confirmProtectedAttrSelection() {
  updateProtectedAttrDisplay();

  closeAttrModal();

  handleProtectedAttributesChange();
}

function confirmTargetAttrSelection() {
  updateTargetAttrDisplay();

  closeTargetModal();

  handleTargetAttributesChange();
}
    
async function handleProtectedAttributesChange() {
  const selectedProtectedAttrs = getSelectedProtectedAttributes();
  const selectedTargetAttrs = getSelectedTargetAttributes();

  displayBiasOverviewLoad();
  
  if (!validateAttributeSelection(selectedProtectedAttrs, selectedTargetAttrs)){
    $('#biasScoreDescription').textContent = 'Invalid Atttibutes Selection';
    return;
  }

  await displayBiasOverview(selectedProtectedAttrs, selectedTargetAttrs);
}

async function handleTargetAttributesChange() {
  const selectedProtectedAttrs = getSelectedProtectedAttributes();
  const selectedTargetAttrs = getSelectedTargetAttributes();

  displayBiasOverviewLoad()

  if (!validateAttributeSelection(selectedProtectedAttrs, selectedTargetAttrs)){
    $('#biasScoreDescription').textContent = 'Invalid Atttibutes Selection';
    return;
  }

  await displayBiasOverview(selectedProtectedAttrs, selectedTargetAttrs);
  await renderFeatureDistributions(selectedTargetAttrs);
}

function validateAttributeSelection(protectedAttrs, targetAttr) {
  const isInvalid = !!(
    targetAttr &&
    Array.isArray(protectedAttrs) &&
    protectedAttrs.includes(targetAttr)
  );

  if (isInvalid) {
    alert("Target attribute cannot be one of the protected attributes.");
    updateProcessStatus('Warning');
    $('#btnRun').disabled = true;

    return false;
  } else {
    updateProcessStatus('Ready');
    $('#btnRun').disabled = false;

    return true;
  }
}