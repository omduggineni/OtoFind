/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.html"
  ],
  theme: {
    screens: {
      'mobile-small': '260px',
      'mobile': '320px',
      'mobile-large': '400px',
      'desktop-sm': '768px',
      'desktop': '1024px',
      'desktop-lg': '1280px',
      'desktop-xl': '1536px',
    }
  }
}
