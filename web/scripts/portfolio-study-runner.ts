import { readFileSync, writeFileSync } from "node:fs";
import { runPortfolioStudy } from "../src/lib/portfolioStudy";

const [input, output] = process.argv.slice(2);
if (!input || !output) throw new Error("Usage: portfolio-study-runner input.json output.json");
const { data, options } = JSON.parse(readFileSync(input, "utf8"));
writeFileSync(output, JSON.stringify(runPortfolioStudy(data, options)));
