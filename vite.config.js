import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  return {
    plugins: [react()],
    define: {
      // Mapping biến môi trường Vercel (VITE_API_KEY) vào process.env.API_KEY để code cũ không bị lỗi
      'process.env.API_KEY': JSON.stringify(env.VITE_API_KEY)
    }
  }
})