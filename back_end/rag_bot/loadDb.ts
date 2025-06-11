import { DataAPIClient } from "@datastax/astra-db-ts";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAI } from "@google/generative-ai";

import "dotenv/config";

type SimilarityMetric = "dot_product" | "cosine" | "euclidean";

const {
    ASTRA_DB_NAMESPACE,
    ASTRA_DB_COLLECTION,
    ASTRA_DB_API_ENDPOINT,
    ASTRA_DB_APPLICATION_TOKEN,
    GEMINI_API_KEY
} = process.env;

const disease_data = [
    "https://www.intechopen.com/chapters/1160040"
];

const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db = client.db(ASTRA_DB_API_ENDPOINT, { namespace: ASTRA_DB_NAMESPACE });

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 512,
    chunkOverlap: 100
});

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY as string);
const model = genAI.getGenerativeModel({ model: "embedding-001" });

const createCollection = async (similarityMetric: SimilarityMetric = "dot_product") => {
    try {
        const res = await db.createCollection(ASTRA_DB_COLLECTION, {
            vector: {
                dimension: 768,
                metric: similarityMetric
            }
        });
        console.log("Collection created:", res);
    } catch (error) {
        console.error("Error creating collection:", error);
    }
};


const loadSampleData = async () => {
    try {
        const collection = await db.collection(ASTRA_DB_COLLECTION);

        for (const url of disease_data) {
            console.log(`Scraping content from: ${url}`);
            const content = await scrapePage(url);
            const chunks = await splitter.splitText(content);

            for (const chunk of chunks) {
                console.log("Processing chunk:", chunk.substring(0, 50) + "...");

                // Generate embeddings using correct format
                const response = await model.embedContent([{ text: chunk }]);

                const vector = response.embedding?.values;

                if (!vector) {
                    console.warn("Skipping embedding: No vector generated.");
                    continue;
                }

                // Insert chunk with its embedding into the database
                await collection.insertOne({
                    text: chunk,
                    vector: vector
                });

                console.log("Inserted chunk into DB.");
            }
        }
        console.log("All data loaded successfully.");
    } catch (error) {
        console.error("Error loading data:", error);
    }
};

const scrapePage = async (url: string): Promise<string> => {
    try {
        const loader = new PuppeteerWebBaseLoader(url, {
            launchOptions: {
                headless: true
            },
            gotoOptions: {
                waitUntil: "domcontentloaded"
            },
            evaluate: async (page, browser) => {
                const result = await page.evaluate(() => document.body.innerText);
                await browser.close();
                return result;
            }
        });
        return await loader.scrape();
    } catch (error) {
        console.error("Error scraping page:", error);
        return "";
    }
};

createCollection()
    .then(() => loadSampleData())
    .catch((err) => console.error("Error initializing system:", err));
