import praw
from dotenv import load_dotenv
import os

load_dotenv()

# Authenticate
reddit = praw.Reddit(
    client_id=os.getenv("clientID"),
    client_secret=os.getenv("clientSecret"),
    user_agent="csds312 by u/Initial-Adventurous",
    # username="YOUR_USERNAME",       # Only required for read/write actions
    # password="YOUR_PASSWORD"        # Only required for read/write actions
)

# Example: Read posts from a subreddit
# subreddit: String name of the subreddit we're searching
# keyWords: Key words of posts were searching for
# sortBy: How to sort posts "Hot", "Top", "New", "Relevance"
# postLimit: Number of posts grabbed
# commentAmount: Number of comments grabbed
# commentLength: Number of characters per comment
def search_reddit(subreddit, keyWords, sortBy, postLimit, commentAmount, commentLength):
    subreddit = reddit.subreddit(subreddit)
    for post in subreddit.search(keyWords, sort = sortBy, limit = postLimit):
        print(f"[{post.score}] {post.title} (r/{post.subreddit})")
        
        post.comments.replace_more(limit=0)  # Removes "load more" placeholders
        for comment in post.comments[:commentAmount]:  # Just get the first 3 comments
            print(f"  - {comment.body[:commentLength]}")  # Print the first 150 characters

search_reddit("movies", "minecraft movie", "relevance", 10, 3, 150)