import instaloader
import json

# Create an instance of the Instaloader class
loader = instaloader.Instaloader()

# Load the profile of the user
profile = instaloader.Profile.from_username(loader.context, "ishann_g")

# Extract profile metadata into separate variables
username = profile.username
full_name = profile.full_name
biography = profile.biography
followers_count = profile.followers
following_count = profile.followees

# Additional features
has_profile_picture = bool(profile.profile_pic_url)  # 0 or 1 (1 if there's a profile picture, 0 otherwise)

numerical_chars_in_username = sum(c.isdigit() for c in username)
username_length = len(username)
ratio_numerical_chars_in_username = numerical_chars_in_username / username_length  # Decimal format

full_name_tokens = len(full_name.split())
numerical_chars_in_full_name = sum(c.isdigit() for c in full_name)
full_name_length = len(full_name)
ratio_numerical_chars_in_full_name = numerical_chars_in_full_name / full_name_length  # Float format

same_username_and_full_name = int(username.lower() == full_name.lower())  # 0 or 1

bio_length = len(biography)  # Number of characters in the biography

has_external_url = int(bool(profile.external_url))  # 0 if no external URL, 1 if there is one

is_private = int(profile.is_private)  # 0 if not private, 1 if private

num_posts = profile.mediacount  # Number of posts

print(f"Username: {username}")
print(f"Full Name: {full_name}")
print(f"Biography: {biography}")
print(f"Followers Count: {followers_count}")
print(f"Following Count: {following_count}")

print(f"Has Profile Picture: {has_profile_picture}")
print(f"Ratio of Numerical Chars in Username: {ratio_numerical_chars_in_username}")
print(f"Full Name in Word Tokens: {full_name_tokens}")
print(f"Ratio of Numerical Chars in Full Name: {ratio_numerical_chars_in_full_name}")
print(f"Username and Full Name Literally the Same: {same_username_and_full_name}")
print(f"Bio Length (in characters): {bio_length}")
print(f"Has External URL: {has_external_url}")
print(f"Is Private: {is_private}")
print(f"Number of Posts: {num_posts}")



