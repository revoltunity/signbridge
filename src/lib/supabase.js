import { createClient } from '@supabase/supabase-js';

const url = import.meta.env.VITE_SUPABASE_URL;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const isSupabaseConfigured = Boolean(url && anonKey);

export const supabase = isSupabaseConfigured
  ? createClient(url, anonKey)
  : null;

// Example for later:
// await supabase.from('translations').insert({
//   direction: 'text_to_sign',
//   source_text: inputText,
//   translated_text: outputText
// });