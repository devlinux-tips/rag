/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL?: string
  readonly VITE_WEB_API_URL?: string
  readonly VITE_WEB_API_PORT?: string
  readonly VITE_API_HOST?: string
  readonly MODE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}