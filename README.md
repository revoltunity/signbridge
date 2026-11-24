# SignBridge — starter project

React + Vite starter for a two-way Indonesian/sign-language translation website.

## 1. Requirements
- Node.js LTS (recommended)
- npm
- Chrome or Edge for browser speech recognition testing

## 2. Run locally
```bash
npm install
cp .env.example .env.local
npm run dev
```
Open the local URL printed by Vite (usually http://localhost:5173).

On Windows PowerShell, copy the environment template with:
```powershell
Copy-Item .env.example .env.local
```

## 3. Supabase setup
1. Create a project at https://supabase.com/dashboard.
2. Open Project Settings → API.
3. Copy the Project URL and `anon` / publishable key into `.env.local`.
4. Restart `npm run dev`.

Use only the browser-safe anon/publishable key in `VITE_` variables. Never put the `service_role` key in frontend code.

Suggested first table (run in Supabase SQL Editor):
```sql
create table public.translations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete set null,
  direction text not null check (direction in ('text_to_sign', 'sign_to_text')),
  source_text text,
  translated_text text,
  language_standard text not null default 'SIBI',
  created_at timestamptz not null default now()
);

alter table public.translations enable row level security;

create policy "Users can read their own translations"
on public.translations for select
to authenticated
using (auth.uid() = user_id);

create policy "Users can insert their own translations"
on public.translations for insert
to authenticated
with check (auth.uid() = user_id);
```
For anonymous use, do not add permissive public write policies without a clear abuse and privacy plan.

## 4. What is currently functional
- Responsive translator UI
- Text input
- Browser speech-to-text where supported
- Browser text-to-speech
- Camera permission and live preview
- Supabase client configuration indicator

## 5. Important: what is still a prototype
The current "translation" output is a placeholder. Camera preview does not recognize signs yet. The avatar is a visual placeholder, not a real sign-language animation system.

A real implementation needs:
1. Confirm one standard: SIBI or BISINDO (they are not interchangeable).
2. Obtain or create a licensed, consented, labeled dataset.
3. Implement MediaPipe landmark extraction (hands; likely pose/face context too).
4. Train/evaluate a temporal classifier for sign sequences, not just static hand poses.
5. Build a translation layer from recognized signs to natural Indonesian.
6. Build a verified sign animation library and retarget animations to a 3D avatar.

## 6. Deploy to Vercel
1. Push this project to a GitHub repository.
2. In Vercel, choose Add New → Project and import the repository.
3. Framework preset: Vite. Build command: `npm run build`. Output directory: `dist`.
4. Add `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` in Vercel Project Settings → Environment Variables.
5. Deploy. Redeploy after changing environment variables.

Camera and microphone access require HTTPS in production; Vercel provides HTTPS.

## 7. Suggested next development milestone
Integrate `@mediapipe/tasks-vision` in `src/lib/signRecognition.js`, render landmarks on a canvas overlay, and validate tracking before training or connecting a sign classifier.
