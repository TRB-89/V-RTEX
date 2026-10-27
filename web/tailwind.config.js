/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Void Black primary tones
        void: {
          DEFAULT: '#08080C',
          deep: '#050508',
          surface: '#111216',
          elevated: '#18191E',
        },
        // Charcoal intermediate tones
        charcoal: {
          DEFAULT: '#1B1C20',
          light: '#25262B',
          border: '#2A2B32',
          muted: '#363840',
        },
        // Steel grey architectural lines
        steel: {
          DEFAULT: '#48494D',
          light: '#5E6066',
          dim: '#323338',
        },
        // Ash grey editorial text
        ash: {
          DEFAULT: '#999A9E',
          light: '#B2B3B8',
          dark: '#717277',
        },
        // Silver white high-contrast editorial text
        silver: {
          DEFAULT: '#E7E8EA',
          bright: '#F4F5F7',
          muted: '#C4C5CA',
        },
        // Restrained electric blue (2% accent)
        electric: {
          DEFAULT: '#2514ED',
          glow: '#3828FA',
          deep: '#180AB8',
        },
        // Backward-compatible aliases mapped strictly to monochrome palette
        obsidian: {
          DEFAULT: '#08080C',
          deep: '#050508',
          surface: '#111216',
          elevated: '#18191E',
        },
        graphite: {
          DEFAULT: '#1B1C20',
          light: '#25262B',
          border: '#2A2B32',
          muted: '#363840',
        },
        violet: {
          muted: '#999A9E', // mapped to ash grey to eliminate purple hue
          dim: '#5E6066',
        },
        chrome: {
          DEFAULT: '#E7E8EA',
          muted: '#999A9E',
          bright: '#F4F5F7',
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
        display: ['"Inter"', 'system-ui', '-apple-system', 'sans-serif'],
        sans: ['"Inter"', 'system-ui', '-apple-system', 'sans-serif'],
      },
      letterSpacing: {
        'tightest': '-0.05em',
        'tighter': '-0.03em',
        'editorial': '0.28em',
        'wide-editorial': '0.35em',
      },
      backgroundImage: {
        'grain': "radial-gradient(circle at 50% 50%, rgba(231, 232, 234, 0.02) 0%, transparent 80%)",
        'atmospheric-gradient': "radial-gradient(ellipse at 50% 15%, rgba(153, 154, 158, 0.08) 0%, rgba(8, 8, 12, 0.98) 75%)",
      }
    },
  },
  plugins: [],
}
