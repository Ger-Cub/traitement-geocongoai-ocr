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
        type: "pdf_url",
        pdfUrl: url
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

app.listen(PORT, () => {
  console.log(`🚀 OCR API running on port http://localhost:${PORT}`);
});
