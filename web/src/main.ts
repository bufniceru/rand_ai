import { createApp } from "vue";
import App from "./App.vue";
import LastSeenDialogApp from "./LastSeenDialogApp.vue";
import LastSeenGapDialogApp from "./LastSeenGapDialogApp.vue";
import CombinedPredictionDialogApp from "./CombinedPredictionDialogApp.vue";
import PossibleDrawDialogApp from "./PossibleDrawDialogApp.vue";
import DrawEditorDialogApp from "./DrawEditorDialogApp.vue";
import "./styles.css";

const windowKind = new URLSearchParams(window.location.search).get("window");
const rootComponent =
  windowKind === "last-seen"
    ? LastSeenDialogApp
    : windowKind === "last-seen-gap"
      ? LastSeenGapDialogApp
      : windowKind === "combined-prediction"
        ? CombinedPredictionDialogApp
        : windowKind === "possible-draw"
          ? PossibleDrawDialogApp
          : windowKind === "draw-editor"
            ? DrawEditorDialogApp
        : App;
createApp(rootComponent).mount("#app");
