<template>
  <form @submit.prevent="analyzeSubreddit" class="form-container">
    <div class="form-group">
      <label>Subreddit:</label>
      <input v-model="subreddit" placeholder="e.g., Iphone" required />
    </div>
    <div class="form-group">
      <label>Number of Posts:</label>
      <input type="number" v-model.number="top_n" min="1" max="100" />
    </div>
    <button type="submit" :disabled="loading">Analyze</button>

    <div v-if="loading" class="loading">
      <p>Analyzing... Please wait.</p>
      <progress max="100"></progress>
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
const loading = ref(false)
const result = ref(null)
const error = ref('')

async function analyzeSubreddit() {
  error.value = ''
  result.value = null
  loading.value = true

  try {
    const data = await analyzeSubredditAPI(subreddit.value, top_n.value)
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
  display: flex;
  flex-direction: column;
  gap: 1rem;

  .form-group {
    display: flex;
    flex-direction: column;
  }

  button {
    padding: 0.5rem 1rem;
    width: fit-content;
  }

  .loading {
    margin-top: 1rem;
  }

  .error {
    color: red;
  }

  .result-container {
    margin-top: 2rem;
    img {
      max-width: 100%;
      height: auto;
      margin-top: 1rem;
    }
  }
}
</style>
