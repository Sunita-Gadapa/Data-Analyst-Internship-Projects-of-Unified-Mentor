CREATE DATABASE  IF NOT EXISTS `InstagramFakeDetection_db`;
DROP DATABASE IF EXISTS `InstagramFakeDetection_db`;
use InstagramFakeDetection_db;

-- Sample records
SELECT * FROM instagram_accounts; 

-- Quick row count
SELECT COUNT(*) AS rows_loaded FROM instagram_accounts;

-- Null checks (should be few or zero)
SELECT
  SUM(profile_pic IS NULL) AS null_profile_pic,
  SUM(username_num_ratio IS NULL) AS null_username_num_ratio,
  SUM(fol_ratio IS NULL) AS null_fol_ratio
FROM instagram_accounts;

-- Range/summary sanity
SELECT
  MIN(followers) AS min_followers, MAX(followers) AS max_followers,
  MIN(follows) AS min_follows,   MAX(follows) AS max_follows,
  MIN(fol_ratio) AS min_folratio,   MAX(fol_ratio) AS max_folratio
FROM instagram_accounts;

-- Class counts
SELECT
  SUM(fake=1) AS fake_cnt,
  SUM(fake=0) AS genuine_cnt,
  COUNT(*) AS total
FROM instagram_accounts;

-- Profile pic distribution by class
SELECT profile_pic, fake, COUNT(*) AS cnt
FROM instagram_accounts
GROUP BY profile_pic, fake
ORDER BY profile_pic, fake;

-- Follower-following ratios (outliers)
SELECT account_id, followers, follows, fol_ratio, fake
FROM instagram_accounts
ORDER BY fol_ratio DESC
LIMIT 20;

-- Username pattern signals by class
SELECT fake,
       AVG(username_num_ratio) AS avg_un_ratio,
       AVG(fullname_num_ratio) AS avg_fn_ratio,
       AVG(fullname_words)     AS avg_words
FROM instagram_accounts
GROUP BY fake;

-- Private vs public by class
SELECT is_private, fake, COUNT(*) AS cnt
FROM instagram_accounts
GROUP BY is_private, fake
ORDER BY is_private, fake;

-- Activity stats by class
SELECT fake,
       AVG(posts) AS avg_posts,
       AVG(followers) AS avg_followers,
       AVG(follows) AS avg_follows,
       AVG(fol_ratio) AS avg_ratio
FROM instagram_accounts
GROUP BY fake;
