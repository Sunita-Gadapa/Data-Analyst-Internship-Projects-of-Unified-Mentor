CREATE DATABASE  IF NOT EXISTS `Netflix_Analytics`;
DROP DATABASE IF EXISTS `Netflix_Analytics`;
 
Use Netflix_Analytics;

select * from titles;
select * from genres;

Select distinct(count(genre)) from genres;
Select distinct(count(listed_in)) from titles;

-- 1) Content dominance: Movie vs TV Show
SELECT type, COUNT(*) AS Type_count
FROM titles
GROUP BY type
ORDER BY Type_count DESC;

-- 2) Yearly trend of additions
SELECT year(date_added) AS year_added, COUNT(*) AS titles_added
FROM titles
GROUP BY year_added
ORDER BY year_added;

-- 3) Top 10 countries
SELECT country, COUNT(*) AS country_count
FROM titles
GROUP BY country
ORDER BY country_count DESC
LIMIT 10;

-- 4) Ratings composition
SELECT rating, COUNT(*) AS rating_count
FROM titles
GROUP BY rating
ORDER BY rating_count DESC;

-- 5) Month seasonality (overall)
SELECT month(date_added) AS month_added, COUNT(*) AS month_count
FROM titles
GROUP BY month_added
ORDER BY month_added;

-- 6) Top genres (using exploded table)
SELECT genre, COUNT(*) AS genre_count
FROM genres
GROUP BY genre
ORDER BY genre_count DESC
LIMIT 15;

-- 7) Country-genre matrix: which genres thrive where (sample: India/US)
SELECT t.country, g.genre, COUNT(*) AS cnt
FROM titles t
JOIN genres g ON t.show_id = g.show_id
WHERE t.country IN ('India','United States')
GROUP BY t.country, g.genre
ORDER BY t.country, cnt DESC;

-- 8) Directors with most titles
SELECT director, COUNT(*) AS directors_count
FROM titles
WHERE director <> 'Not Given'
GROUP BY director
ORDER BY directors_count DESC
LIMIT 15;

-- 9) Type vs rating: audience targeting patterns
SELECT type, rating, COUNT(*) AS cnt
FROM titles
GROUP BY type, rating
ORDER BY type, cnt DESC;
