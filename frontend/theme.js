// Theme Toggle
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme') || 'light';
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Update icon visibility
    const lightIcon = $('.theme-icon--light');
    const darkIcon = $('.theme-icon--dark');
    if (newTheme === 'dark') {
    lightIcon.style.display = 'none';
    darkIcon.style.display = 'block';
    } else {
    lightIcon.style.display = 'block';
    darkIcon.style.display = 'none';
    }
}

// Initialize theme
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    const lightIcon = $('.theme-icon--light');
    const darkIcon = $('.theme-icon--dark');
    if (savedTheme === 'dark') {
    lightIcon.style.display = 'none';
    darkIcon.style.display = 'block';
    } else {
    lightIcon.style.display = 'block';
    darkIcon.style.display = 'none';
    }
}

$('#btnTheme').addEventListener('click', toggleTheme);