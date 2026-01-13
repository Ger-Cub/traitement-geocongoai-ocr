import express, { Request, Response } from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { Mistral } from '@mistralai/mistralai';
import { z } from 'zod';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const apiKey = process.env.MISTRAL_API_KEY;

if (!apiKey) {
  console.error('❌ MISTRAL_API_KEY is missing in environment variables.');
  process.exit(1);
}

const client = new Mistral({ apiKey });

// Validation Schema
const OcrRequestSchema = z.object({
  type: z.enum(['pdf', 'image']),
  url: z.string().url(),
  metadata: z.object({
    source: z.string().optional(),
    document_id: z.string().optional(),
  }).optional(),
});

app.post('/ocr', async (req: Request, res: Response) => {
  try {
    // 1. Validate Input
    const parseResult = OcrRequestSchema.safeParse(req.body);
    if (!parseResult.success) {
      res.status(400).json({
        status: 'error',
        message: 'Invalid input',
        errors: parseResult.error.issues,
      });
      return;
    }

    const { type, url } = parseResult.data;
    console.info(`[OCR] Processing ${type} from ${url}`);

    // 2. Prepare Mistral Request
    let documentPayload: any;
    if (type === 'pdf') {
      documentPayload = {
        type: "document_url",
        documentUrl: url
      };
    } else {
      // Mistral uses image_url for images 
      documentPayload = {
        type: "image_url",
        imageUrl: url
      };
    }

    // 3. Call Mistral OCR
    const ocrResponse = await client.ocr.process({
      model: "mistral-ocr-latest",
      document: documentPayload,
      includeImageBase64: true,
    });

    // 4. Transform Response (Normalized)
    const pages = ocrResponse.pages.map((page: any, index: number) => {
      // Mistral pages usually have 'index', 'markdown', and potentially 'images' if requested.
      const imgBase64 = (page.images && page.images.length > 0) ? page.images[0].base64 : null;

      return {
        page: page.index !== undefined ? page.index + 1 : index + 1,
        text: page.markdown, // Using markdown as text representation
        markdown: page.markdown,
        image_base64: imgBase64
      };
    });

    // 5. Return Success
    res.json({
      status: 'success',
      pages,
      // raw: ocrResponse 
    });

  } catch (error: any) {
    console.error('[OCR] Error processing document:', error);

    // Handle Mistral specific errors or generic errors
    const statusCode = error.statusCode || 500;
    const message = error.message || 'Internal Server Error';

    res.status(statusCode).json({
      status: 'error',
      message
    });
  }
});

app.post('/ocr-maps', async (req: Request, res: Response) => {
  try {
    // 1. Validate Input
    const parseResult = OcrRequestSchema.safeParse(req.body);
    if (!parseResult.success) {
      res.status(400).json({
        status: 'error',
        message: 'Invalid input',
        errors: parseResult.error.issues,
      });
      return;
    }

    const { type, url } = parseResult.data;
    console.info(`[OCR-MAPS] Processing ${type} from ${url}`);

    // 2. Fetch and convert document to base64
    const docResponse = await fetch(url);
    if (!docResponse.ok) {
      throw new Error(`Failed to fetch document: ${docResponse.statusText}`);
    }
    const arrayBuffer = await docResponse.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    const base64Doc = buffer.toString('base64');
    const mimeType = docResponse.headers.get('content-type') || (type === 'pdf' ? 'application/pdf' : 'image/jpeg');
    const dataUrl = `data:${mimeType};base64,${base64Doc}`;

    // 3. For PDFs, we need to use OCR first to extract images, then analyze with Vision
    // For images, we can directly use Vision
    let pages: any[] = [];

    if (type === 'pdf') {
      // Use OCR to extract images from PDF
      const ocrResponse = await client.ocr.process({
        model: "mistral-ocr-latest",
        document: {
          type: "document_url",
          documentUrl: url
        },
        includeImageBase64: true,
      });

      // Now analyze each extracted image with Pixtral for detailed description
      for (let i = 0; i < ocrResponse.pages.length; i++) {
        const page = ocrResponse.pages[i];
        const imgBase64 = (page.images && page.images.length > 0) ? (page.images[0] as any).base64 : null;

        let description = page.markdown || ""; // Fallback to OCR text

        if (imgBase64) {
          // Use Vision model for detailed analysis
          const imageDataUrl = `data:image/jpeg;base64,${imgBase64}`;

          try {
            const chatResponse = await client.chat.complete({
              model: "pixtral-12b-2409",
              messages: [
                {
                  role: "user",
                  content: [
                    { type: "text", text: "Agis comme un expert en géologie. Ne te contente pas d'extraire le texte. Décris visuellement la carte géologique : identifie les couleurs de la légende et associe-les aux formations sur la carte. Analyse les symboles miniers (gisements, forages) et les structures tectoniques visibles. Décris l'échelle, l'orientation, et toute information géographique pertinente." },
                    { type: "image_url", imageUrl: imageDataUrl }
                  ]
                }
              ] as any,
            });

            const content = chatResponse.choices && chatResponse.choices.length > 0
              ? chatResponse.choices[0].message.content
              : null;
            description = (typeof content === 'string' ? content : description);
          } catch (visionError) {
            console.error(`[OCR-MAPS] Vision analysis failed for page ${i + 1}, using OCR text:`, visionError);
          }
        }

        pages.push({
          page: page.index !== undefined ? page.index + 1 : i + 1,
          text: description,
          markdown: description,
          image_base64: imgBase64,
          data_url: imgBase64 ? `data:image/jpeg;base64,${imgBase64}` : null
        });
      }
    } else {
      // For images, directly use Vision model
      const chatResponse = await client.chat.complete({
        model: "pixtral-12b-2409",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "Agis comme un expert en géologie. Ne te contente pas d'extraire le texte. Décris visuellement la carte géologique : identifie les couleurs de la légende et associe-les aux formations sur la carte. Analyse les symboles miniers (gisements, forages) et les structures tectoniques visibles. Décris l'échelle, l'orientation, et toute information géographique pertinente." },
              { type: "image_url", imageUrl: dataUrl }
            ]
          }
        ] as any,
      });

      const content = chatResponse.choices && chatResponse.choices.length > 0
        ? chatResponse.choices[0].message.content
        : null;
      const description = typeof content === 'string' ? content
        : "No description generated.";

      pages.push({
        page: 1,
        text: description,
        markdown: description,
        image_base64: base64Doc,
        data_url: dataUrl
      });
    }

    // 4. Return Success with detailed descriptions and data URLs
    res.json({
      status: 'success',
      pages,
      document_data_url: dataUrl // Original document data URL
    });

  } catch (error: any) {
    console.error('[OCR-MAPS] Error processing map:', error);
    const statusCode = error.statusCode || 500;
    const message = error.message || 'Internal Server Error';

    res.status(statusCode).json({
      status: 'error',
      message
    });
  }
});

app.post('/ocr-rocks', async (req: Request, res: Response) => {
  try {
    // 1. Validate Input
    const parseResult = OcrRequestSchema.safeParse(req.body);
    if (!parseResult.success) {
      res.status(400).json({
        status: 'error',
        message: 'Invalid input',
        errors: parseResult.error.issues,
      });
      return;
    }

    const { type, url } = parseResult.data;
    console.info(`[OCR-ROCKS] Processing ${type} from ${url}`);

    // 2. Fetch Image to get Base64 (Required for Data URL return and reliable Vision API usage)
    const imageResponse = await fetch(url);
    if (!imageResponse.ok) {
      throw new Error(`Failed to fetch image: ${imageResponse.statusText}`);
    }
    const arrayBuffer = await imageResponse.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    const base64Image = buffer.toString('base64');
    const mimeType = imageResponse.headers.get('content-type') || 'image/jpeg';
    const dataUrl = `data:${mimeType};base64,${base64Image}`;

    // 3. Call Mistral Vision (Pixtral) for meaningful description
    // OCR models often return empty text for nature images without text.
    // Pixtral is better for "describing" the rock.
    const chatResponse = await client.chat.complete({
      model: "pixtral-12b-2409", // Using a Pixtral model for vision capabilities
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Agis comme un expert géologue et pétrographe. Analyse visuellement cette image de roche. Identifie le type de roche (ignée, sédimentaire, métamorphique), sa texture, sa couleur, sa granulométrie et les minéraux visibles. Décris ses caractéristiques structurelles (veines, fractures, porosité). Estime sa composition minéralogique probable et propose une classification précise." },
            { type: "image_url", imageUrl: dataUrl } // Use dataUrl directly
          ]
        }
      ] as any, // Cast to any because TS definition might strictly expect text-only content in older types
    });

    const description = chatResponse.choices && chatResponse.choices.length > 0
      ? chatResponse.choices[0].message.content
      : "No description generated.";

    // 4. Return Success with Data URL and Description
    res.json({
      status: 'success',
      data_url: dataUrl, // Returning the data URL as requested
      description: description,
      // Keeping a unified structure if the frontend expects 'pages', but 'description' is primary here.
      pages: [{
        page: 1,
        text: description,
        markdown: description,
        image_base64: base64Image // Raw base64 if needed separately
      }]
    });

  } catch (error: any) {
    console.error('[OCR-ROCKS] Error processing rock image:', error);
    const statusCode = error.statusCode || 500;
    const message = error.message || 'Internal Server Error';

    res.status(statusCode).json({
      status: 'error',
      message
    });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 OCR API running on port http://localhost:${PORT}`);
});
