import { config } from "dotenv";
config({ path: "web/.env" });
import { stitch } from "@google/stitch-sdk";
import { writeFileSync } from "fs";

const OUTPUT_FILE = "docs/index.html";
const PLACEHOLDER_URL = "MODAL_PREDICT_URL";

const PROMPT = `
Design a professional, modern dog breed classifier web application page.

Layout and style:
- Dark background (#0f0f0f or similar deep dark)
- Clean white/light gray text
- Centered content with max-width container
- Fully mobile responsive

Header:
- Large bold title: "Dog Breed Classifier"
- Subtitle: "Upload a photo of your dog to identify its breed"
- A small dog paw emoji or icon near the title

Upload section:
- A large rectangular drag-and-drop zone with a dashed border (rounded corners)
- Dog paw or upload icon in the center of the zone
- Text inside: "Drag & drop your dog photo here" and below it "or click to browse"
- The entire zone is clickable to open a file picker
- Accepts image files only (jpg, png, webp)
- Below the zone: a clearly visible "Identify Breed" button (accent color, rounded)

Results section (hidden by default, shown after upload):
- Two-column layout on desktop, stacked on mobile
- Left column: the uploaded dog photo (displayed as a preview thumbnail, nicely rounded corners)
- Right column: "Results" heading, then a list of top 3 predictions
- Each prediction row shows:
  - Breed name (formatted nicely, e.g. "Golden Retriever" not "golden_retriever")
  - Confidence percentage (e.g. "94.2%")
  - A horizontal progress bar showing the confidence level
  - First result is highlighted as the top prediction
- A loading spinner shown while waiting for results

Footer:
- Small text: "Powered by MobileNet V2 · Trained on 120 dog breeds"

The fetch call to send the image should POST to: ${PLACEHOLDER_URL}
Use FormData to send the image file with field name "file".
Parse the JSON response which has shape: { predictions: [{breed, confidence}, ...] }
`;

console.log("Creating Stitch project...");
const project = await stitch.createProject("Dog Breed Classifier");
console.log("Project ID:", project.id);

console.log("Generating UI screen (this may take 30-60 seconds)...");
const raw = await stitch.callTool("generate_screen_from_text", {
  projectId: project.id,
  prompt: PROMPT,
});

// The SDK hardcodes index [0] but the design component is at a different index.
// Find the component that actually contains the design + screen data.
const designComponent = raw.outputComponents.find((c) => c.design?.screens?.[0]?.htmlCode?.downloadUrl);
if (!designComponent) {
  console.error("Could not find a screen with HTML in the response.");
  console.error("Components:", raw.outputComponents.map((c) => Object.keys(c)));
  process.exit(1);
}

const htmlUrl = designComponent.design.screens[0].htmlCode.downloadUrl;
console.log("Downloading generated HTML...");
const response = await fetch(htmlUrl);
const html = await response.text();

writeFileSync(OUTPUT_FILE, html, "utf-8");
console.log(`\nDone! HTML saved to: ${OUTPUT_FILE}`);
console.log(`\nNext: replace "${PLACEHOLDER_URL}" in ${OUTPUT_FILE} with your Modal endpoint URL.`);
