import praw
from dotenv import load_dotenv
import os
import psycopg2
from supabase import create_client
load_dotenv()

# Authenticate
reddit = praw.Reddit(
    client_id=os.getenv("clientID"),
    client_secret=os.getenv("clientSecret"),
    user_agent="csds312 by u/Initial-Adventurous",
    # username="YOUR_USERNAME",       # Only required for read/write actions
    # password="YOUR_PASSWORD"        # Only required for read/write actions
)

url = "https://flpnzndwioffpeyozign.supabase.co"
key ="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZscG56bmR3aW9mZnBleW96aWduIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUzNzAyOTEsImV4cCI6MjA2MDk0NjI5MX0.bEIk3wA28khyHaa2hw6BhtX3MHPVKLL57qX3HSCu8E0"
supabase = create_client(url, key)

# Example: Read posts from a subreddit
# subreddit: String name of the subreddit we're searching
# keyWords: Key words of posts were searching for
# sortBy: How to sort posts "Hot", "Top", "New", "Relevance"
# postLimit: Number of posts grabbed
# commentAmount: Number of comments grabbed
# commentLength: Number of characters per comment
def search_reddit(subreddit_name, keyWords, sortBy, postLimit, commentAmount, commentLength):
    # PostgreSQL connection
    conn = psycopg2.connect(
        dbname= os.getenv("dbname"),
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=os.getenv("port")
    )
    cur = conn.cursor()

    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.search(keyWords, sort=sortBy, limit=postLimit):
        print(f"[{post.score}] {post.title} (r/{post.subreddit})")

        # Insert post into DB
        cur.execute("""
            INSERT INTO posts (id, subreddit, title, score)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
        """, (post.id, str(post.subreddit), post.title, post.score))

        post.comments.replace_more(limit=0)
        for comment in post.comments[:commentAmount]:
            body_trimmed = comment.body[:commentLength]
            print(f"  - {body_trimmed}")

            # Insert comment into DB
            cur.execute("""
                INSERT INTO comments (id, post_id, body)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO NOTHING;
            """, (comment.id, post.id, body_trimmed))

    conn.commit()
    cur.close()
    conn.close()

#search_reddit("movies", "minecraft movie", "relevance", 10, 3, 150)

def show_reddit_tables():
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname= os.getenv("dbname"),
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=os.getenv("port")
    )
    cur = conn.cursor()

    print("\n📄 Posts Table:")
    cur.execute("SELECT id, subreddit, title, score FROM posts LIMIT 10;")
    posts = cur.fetchall()
    for post in posts:
        print(f"ID: {post[0]} | Subreddit: r/{post[1]} | Score: {post[3]}")
        print(f"Title: {post[2]}\n")

    print("\n💬 Comments Table:")
    cur.execute("SELECT id, post_id, body FROM comments LIMIT 10;")
    comments = cur.fetchall()
    for comment in comments:
        print(f"ID: {comment[0]} | Post ID: {comment[1]}")
        print(f"Comment: {comment[2]}\n")

    cur.close()
    conn.close()
    
show_reddit_tables()