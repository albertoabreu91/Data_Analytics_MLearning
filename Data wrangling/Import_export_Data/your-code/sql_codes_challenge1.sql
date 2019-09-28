-- Sakila challenges -------


SELECT *
FROM film
JOIN film_actor
USING(film_id)

JOIN actor
USING(actor_id)

JOIN film_category
USING(film_id)

JOIN category
USING(category_id);

-- -------------

SELECT film_id, title, c.name, release_year, `first_name`, `last_name`
FROM film as f
JOIN film_actor as fa
USING(film_id)

JOIN actor as a
USING(actor_id)

JOIN film_category as fc
USING(film_id)

JOIN category as c
USING(category_id);

-- -------------

SELECT film_id, title, c.name, rating, release_year, `first_name`, `last_name`
FROM film as f
LEFT JOIN film_actor as fa
USING(film_id)

JOIN actor as a
USING(actor_id)

JOIN film_category as fc
USING(film_id)

JOIN category as c
USING(category_id);

-- -------------

-- List of all the title with their respectives catergory, rating and released year

SELECT title, c.name AS category, rating, release_year
FROM film as f
LEFT JOIN film_actor as fa
USING(film_id)

JOIN actor as a
USING(actor_id)

JOIN film_category as fc
USING(film_id)

JOIN category as c
USING(category_id)

GROUP BY title,Category, rating, release_year
;

-- Which is te most popular category for studios to make

SELECT c.name AS category, COUNT(title) titles_per_category
FROM film as f
LEFT JOIN film_actor as fa
USING(film_id)

JOIN actor as a
USING(actor_id)

JOIN film_category as fc
USING(film_id)

JOIN category as c
USING(category_id)

GROUP BY c.name
ORDER BY titles_per_category DESC;

-- What is the most popular rating between the titles
SELECT DISTINCT rating -- to know how many ratinhs exist
FROM film;


SELECT rating, SUM(rating) as quantity
FROM film
GROUP BY rating
ORDER BY quantity DESC;

-- Which actors have more than 30 movies apperances
SELECT actor_id, first_name, last_name, COUNT(title) movies_apperances
FROM film
JOIN film_actor
USING(film_id)

JOIN actor
USING(actor_id)

GROUP BY actor_id
HAVING movies_apperances > 30
ORDER BY movies_apperances DESC;