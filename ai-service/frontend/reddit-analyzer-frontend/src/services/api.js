import axios from 'axios'


const BASE_URL = import.meta.env.VITE_API_BASE_URL

export async function analyzeSubredditAPI(subreddit, top_n, temperature=0.5, max_tokens=150) {
  try {
    const response = await axios.post(`${BASE_URL}/analyze`, { subreddit, top_n, temperature, max_tokens })
    return response.data
  } catch (err) {
    console.error("Error calling /analyze:", err)
    throw err
  }
}
