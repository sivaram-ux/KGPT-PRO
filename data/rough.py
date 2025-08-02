import praw
from markdownify import markdownify as md
from bs4 import MarkupResemblesLocatorWarning
import warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

reddit = praw.Reddit(
    client_id="DdSgbY_HvICVQfQlldtvCg",
    client_secret="DQfWqzu7CuyBUj5IWtxlijfeeUa_TA",
    user_agent="KGPBot by u/redditterredd"
)

subreddit = reddit.subreddit("iitkgp")

# Fetch top 100 posts of all time
posts = subreddit.top(limit=10, time_filter="all")

allowed_flairs = {
    "Funda :snoo_dealwithit:",
    "KGP News 📰",
    "Subject Reviews &#10002;&#65039;"
}

output_file = "reddit.md"
post_limit = 500  # adjust as needed

# --- Open output .md file ---
with open(output_file, "w", encoding="utf-8") as f:
    f.write("# IITKGP Reddit Archive (Top All Time Posts)\n\n")

    # --- Fetch posts from all time ---
    for post in subreddit.top(limit=post_limit, time_filter="all"):
        flair = post.link_flair_text or ""
        if flair.strip() not in allowed_flairs:
            continue

        post.comments.replace_more(limit=0)
        top_comments = [md(c.body.strip()) for c in post.comments[:10]]

        f.write(f"## {post.title}\n\n")
        f.write(f"**Flair:** {flair}\n\n")
        f.write(f"**Author:** u/{post.author}\n\n")
        f.write(f"[Post Link]({post.url})\n\n")

        f.write("### Description\n")
        f.write(md(post.selftext or "_No description provided._") + "\n\n")

        f.write("### Top 10 Comments\n")
        if top_comments:
            for i, comment in enumerate(top_comments, 1):
                f.write(f"**{i}.** {comment}\n\n")
        else:
            f.write("_No comments available._\n\n")

        f.write("---\n\n")

print(f"✅ Done! Posts saved to '{output_file}'")
    
