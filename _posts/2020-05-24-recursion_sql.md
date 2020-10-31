---
toc: true
layout: post
description: I create a table of actors and their bacon numbers using recursion in SQL. The largest finite bacon number is 13.
categories: [sql, python]
title: Bacon numbers via Recursive SQL
---



## SQL in Python
I ran the SQL commands in Python using the module sqlite3. I used [this article](https://swcarpentry.github.io/sql-novice-survey/10-prog/index.html) to help me set it all up. Below is the code to run queries on the database *movies.db* in python.

```python
    import sqlite3

    def run_sql(query):
        connection = sqlite3.connect('movies.db')
        cursor = connection.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        connection.close()
    
        return results
```



## Understanding the database
The data is in the form of a database, *movies.db*, and was obtained from the online course [CS50](https://cs50.harvard.edu/x/2020/psets/7/movies/), which in turn was obtained from imdb. To find out the structure of the tables in the database, I ran the following code (obtained from [stackoverflow](https://stackoverflow.com/questions/305378/list-of-tables-db-schema-dump-etc-using-the-python-sqlite3-api)).

```python
    def produce_schema():
        sql = "SELECT name FROM sqlite_master WHERE type = 'table';"
        tables = run_sql(sql)
    
        schema = {}
        for table in tables:
            sql = f"SELECT sql FROM sqlite_master WHERE type = 'table' and name = '{table[0]}'"
            results = run_sql(sql)
            print(results[0][0])
            schema[table[0]] = results[0][0]
    
        return schema

    produce_schema()
```

For the task of determining bacon numbers, the relevant tables and columns are:
* `people`, with columns `id` and `name`. Each person has a unique id. There are 1044499 people in the table.
* `stars`, with columns `movie_id` and `person_id`. Each row tells you that the actor with id `person_id` starred in the movie with id `movie_id`.

Kevin Bacon has an id of 102, found by running the query `SELECT id FROM people WHERE name = 'Kevin Bacon' `



## The recursive query
To understand recursion in SQL, I recommend [this guide](https://www.essentialsql.com/recursive-ctes-explained/) which explains recursion and how to use recursion in SQL, and then the [SQLite documentation](https://www.sqlite.org/lang_with.html) to better understand some implementation details and what options are available to you.

The query I created to produce the table of bacon numbers is:
```sql
    WITH RECURSIVE
     costars(id1, id2) AS
     ( 
       SELECT stars1.person_id, stars2.person_id
       FROM stars AS stars1
       JOIN stars AS stars2
       ON stars1.movie_id = stars2.movie_id
     ),
     bacon(id, num) AS
     (
       VALUES(102, 0)
       UNION
       SELECT id2, num+1
       FROM bacon JOIN costars
       ON bacon.id = costars.id1
       WHERE num < 13
     )
     SELECT bacon.id, name, MIN(num)
     FROM bacon JOIN people ON bacon.id = people.id
     GROUP BY bacon.id
     ORDER BY num
```

* The table `costars` lists all pairs of actors that co-starred in a movie. This is obtained by doing a self-join of the stars table.
* The table `bacon` is the main table. `id` is the id of the actor and `num` is the bacon number.
    * It starts off with the entry `(102,0)`.  102 is Kevin Bacon's id and 0 is Kevin Bacon's Bacon number.
    * Then for any entry `(id, num)` already in the table, we add a new row `(id2, num+1)`, whenever `id2` co-starred with `id`.
    * `num < 13` indicates we will only have a maximum bacon number of 13. Without this manual limit, the recursive query would never end. This is because the underlying data is not acyclic: e.g. if a is a co-star with b, then b is a co-star of a. The number 13 was chosen via trial-and-error. The resulting table does not change if I increase the limit further, which implies that the maximum bacon number is 13.
* In the end, I select the relevant data from this recursive construction. Because each actor can appear many times in this construction, I use `GROUP BY` to ensure each actor appears only once. I use `MIN(num)` to select each actor's earliest appearance. 

The two main problems with this query are that:
* It is inefficient. There is huge redundancy as actors appear many times in the recursive construction. I do not think there is a way of avoiding this within SQL.
* I have to know what the maximum Bacon number is for the query to produce a complete list. I found this using trial-and-error.


## Bacon number of 13
By running a simple query, I find there are two people with the maximum bacon number of 13, Javier Ordonez and Kimberly Peters.  Using the program created for [this CS50 AI](https://cs50.harvard.edu/ai/2020/projects/0/degrees/) project, I could find the path from these actors to Kevin Bacon. As expected, it takes 13 steps (always satisfying to see two different programs being consistent!) and they are below. Note they have the same path.

1. Javier Ordonez/Kimberly Peters and Amanda Brass starred in PRND
2. Amanda Brass and Michael Bayouth starred in Park Reverse Neutral Drive, PRND (Director's cut)
3. Michael Bayouth and Brandy Bourdeaux starred in Grease Trek
4. Brandy Bourdeaux and Kim Beuché starred in Murder Inside of Me
5. Kim Beuché and Ed Baccari starred in Island, Alicia
6. Ed Baccari and Aida Angotti starred in Late Watch
7. Aida Angotti and Lamont Copeland starred in Bottom Out
8. Lamont Copeland and Ashley Marie Arnold starred in Eye Was Blind
9. Ashley Marie Arnold and Sid Bernstein starred in The Rodnees: We Mod Like Dat!
10. Sid Bernstein and Chuck Berry starred in The Beatles: The Lost Concert
11. Chuck Berry and Eric Clapton starred in Chuck Berry Hail! Hail! Rock 'n' Roll
12. Eric Clapton and Tom Cruise starred in Close Up
13. Tom Cruise and Kevin Bacon starred in A Few Good Men



