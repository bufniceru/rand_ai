import { createApp } from "vue";
import App from "./App.vue";
import "./styles.css";
import {
  initializeColorTemplateRuntime,
  setActiveColorTemplate,
  validateColorTemplate,
} from "./lib/colorTemplates";

async function bootstrap(): Promise<void> {
  initializeColorTemplateRuntime();
  try {
    const stored = await window.randAiDesktop?.getColorTemplate();
    if (stored) setActiveColorTemplate(validateColorTemplate(stored).template);
  } catch (error) {
    console.warn("Could not restore the saved color template:", error);
  }
  createApp(App).mount("#app");
}

void bootstrap();
