<template>
  <form @submit.prevent="analyzeSubreddit" class="form-container">
    <h1>Subreddit Sentiment Analysis with Mistral 7B</h1>

    <div class="form-group">
      <label>Subreddit:</label>
      <input v-model="subreddit" placeholder="e.g., iPhone" required />
    </div>

    <div class="form-group">
      <label>Number of Posts: {{ top_n }}</label>
      <input type="range" v-model.number="top_n" min="1" max="25" />
    </div>

    <div class="form-group">
      <label>Temperature: {{ temperature.toFixed(2) }}</label>
      <input type="range" v-model.number="temperature" min="0" max="1" step="0.01" />
    </div>

    <div class="form-group">
      <label>Max Tokens: {{ max_tokens }}</label>
      <input type="range" v-model.number="max_tokens" min="50" max="500" step="10" />
    </div>

    <button type="submit" :disabled="loading">Analyze</button>

    <div v-if="loading" class="loading">
      <div class="spinner"></div>
      <p>Analyzing... Please wait.</p>
    </div>

    <div v-if="error" class="error">{{ error }}</div>

    <div v-if="result" class="result-container">
      <h2>Summary</h2>
      <p>{{ result.summary }}</p>

      <h2>Sentiment Graph</h2>
      <img :src="`data:image/png;base64,${result.sentiment_graph}`" alt="Sentiment graph" />
    </div>
  </form>
</template>

<script setup>
import { ref } from 'vue'
import { analyzeSubredditAPI } from '../services/api'

const subreddit = ref('')
const top_n = ref(25)
const temperature = ref(0.5)
const max_tokens = ref(150)
const loading = ref(false)
const result = ref(null)
const error = ref('')

async function analyzeSubreddit() {
  error.value = ''
  result.value = null
  loading.value = true

  try {
    const data = await analyzeSubredditAPI(subreddit.value, top_n.value, temperature.value, max_tokens.value)
    result.value = data
  } catch (err) {
    error.value = err.response?.data?.detail || err.message
  } finally {
    loading.value = false
  }
}
</script>

<style scoped lang="scss">
.form-container {
  max-width: 700px;
  margin: 2rem auto;
  padding: 2rem;
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.1);
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  font-family: 'Inter', sans-serif;

  .title {
    text-align: center;
    font-size: 1.75rem;
    margin-bottom: 1rem;
    color: #4f46e5;
  }

  .form-group {
    display: flex;
    flex-direction: column;

    label {
      font-weight: 600;
      margin-bottom: 0.5rem;
    }

    input[type="text"],
    input[type="number"] {
      padding: 0.5rem 0.75rem;
      border-radius: 6px;
      border: 1px solid #ccc;
      transition: border-color 0.2s;

      &:focus {
        border-color: #4f46e5;
        outline: none;
      }
    }

    input[type="range"] {
      width: 100%;
    }
  }

  button {
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    border: none;
    background-color: #4f46e5;
    color: #fff;
    font-weight: 600;
    cursor: pointer;
    transition: background-color 0.2s;

    &:disabled {
      background-color: #a5b4fc;
      cursor: not-allowed;
    }

    &:hover:not(:disabled) {
      background-color: #4338ca;
    }
  }

  .loading {
    display: flex;
    align-items: center;
    gap: 1rem;

    p {
      margin: 0;
      font-weight: 500;
    }

    .spinner {
      border: 4px solid #f3f3f3;
      border-top: 4px solid #4f46e5;
      border-radius: 50%;
      width: 24px;
      height: 24px;
      animation: spin 1s linear infinite;
    }

    @keyframes spin {
      0% { transform: rotate(0deg);}
      100% { transform: rotate(360deg);}
    }
  }

  .error {
    color: #dc2626;
    font-weight: 500;
  }

  .result-container {
    margin-top: 2rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding: 1rem;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    background-color: #f9fafb;

    h2 {
      margin-bottom: 0.5rem;
    }

    img {
      max-width: 100%;
      height: auto;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
  }
}
</style>
