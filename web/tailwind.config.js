/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        obsidian: {
          DEFAULT: '#0C0D13',
          deep: '#07080B',
          surface: '#12131C',
          elevated: '#171824',
        },
        graphite: {
          DEFAULT: '#191923',
          light: '#232332',
          border: '#2E2E3F',
          muted: '#3C3C50',
        },
        violet: {
          muted: '#77768F',
          dim: '#4F4E66',
        },
        chrome: {
          DEFAULT: '#E7E8EF',
          muted: '#B0B2C4',
          bright: '#F5F6FA',
        },
        electric: {
          DEFAULT: '#2514ED',
          glow: '#3B2AFA',
          deep: '#190BB8',
        },
        status: {
          optimal: '#10B981',
          infeasible: '#EF4444',
          unbounded: '#F59E0B',
          running: '#3B82F6',
        }
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', '"SF Mono"', 'ui-monospace', 'monospace'],
        display: ['"Space Grotesk"', 'CabinetGrotesk', 'system-ui', 'sans-serif'],
        sans: ['"Inter"', 'system-ui', '-apple-system', 'sans-serif'],
      },
      letterSpacing: {
        'tightest': '-0.06em',
        'tighter': '-0.04em',
        'editorial': '0.22em',
      },
      backgroundImage: {
        'grain': "radial-gradient(circle at 50% 50%, rgba(231, 232, 239, 0.03) 0%, transparent 80%)",
      }
    },
  },
  plugins: [],
}
