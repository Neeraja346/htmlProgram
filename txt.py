from ytsum import answer_youtube_question, set_together_api_key

# Set your Together AI API key
set_together_api_key("d6c91aeefbee37454ef64944f8979d5522ebf964cd93ca31b7e672486f452c26")

# Example usage
youtube_url = "https://www.youtube.com/watch?v=8ocanbiSyV4"
query = "What is the main topic of this video?"

result = answer_youtube_question(youtube_url, query)

print(result)