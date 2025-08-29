<template>
  <div id="app" class="app-container">
    <SubredditForm />
  </div>
</template>

<script setup>
import SubredditForm from './components/SubredditForm.vue'
</script>

<style lang="scss">
.app-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
  font-family: Arial, sans-serif;
}
</style>
