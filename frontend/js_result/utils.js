const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

const state = {
    data : JSON.parse(localStorage.getItem("data")),
    protectedAttrs : JSON.parse(localStorage.getItem("protectedAttrs")),
    features : JSON.parse(localStorage.getItem("features"))
}

console.log(state);

function getCaseInsensitive(obj, key) {
  if (!obj || !key) return undefined;

  const foundKey = Object.keys(obj).find(
    k => k.toLowerCase() === key.toLowerCase()
  );

  return obj[foundKey];
}

document.querySelector('.brand__name').addEventListener('click', () => {
  window.location.href = 'http://8.148.159.241:8000/';
});