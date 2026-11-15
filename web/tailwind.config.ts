import type { Config } from "tailwindcss";
import typography from "@tailwindcss/typography";

const v = (name: string) => `var(--${name})`;

const config: Config = {
  darkMode: "class",
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    container: { center: true, padding: "1rem" },
    extend: {
      colors: {
        background: v("background"),
        foreground: v("foreground"),
        card: v("card"),
        muted: { DEFAULT: v("muted"), foreground: v("muted-foreground") },
        subtle: v("subtle"),
        border: v("border"),
        primary: { DEFAULT: v("primary"), foreground: v("primary-foreground") },
        up: v("up-text"),
        down: v("down-text"),
        warning: { DEFAULT: v("warning-bg"), foreground: v("warning-fg") },
        series: { 1: v("series-1"), 2: v("series-2"), 3: v("series-3") },
      },
      fontFamily: {
        sans: ["system-ui", "-apple-system", '"Segoe UI"', "Roboto", "sans-serif"],
      },
      borderRadius: { lg: "0.625rem", md: "0.5rem", sm: "0.375rem" },
    },
  },
  plugins: [typography],
};

export default config;
