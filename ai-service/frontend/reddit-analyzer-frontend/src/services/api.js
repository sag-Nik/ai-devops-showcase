import axios from 'axios'

// Use environment variable
const BASE_URL = import.meta.env.VITE_API_BASE_URL

export async function analyzeSubredditAPI(subreddit, top_n) {
  const response = await axios.post(`${BASE_URL}/analyze`, { subreddit, top_n })
  return response.data
}
