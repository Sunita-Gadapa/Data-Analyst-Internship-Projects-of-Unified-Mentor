CREATE DATABASE IF NOT EXISTS Laptop_pricing; 
USE Laptop_pricing;

Select * from Laptop_prices;

-- Row count 
SELECT COUNT(*) AS row_count FROM Laptop_prices;

-- Duplicate values except id 
SELECT COUNT(*) AS dup_count 
FROM Laptop_prices 
GROUP BY Company, Product, TypeName, Inches, Ram, OS, Weight, Price_euros, Price_inr, 
Screen, ScreenW, ScreenH, Touchscreen, IPSpanel, RetinaDisplay, CPU_company, CPU_freq, 
CPU_model, PrimaryStorage, SecondaryStorage, PrimaryStorageType, SecondaryStorageType, GPU_company, GPU_model 
HAVING COUNT(*) > 1;

-- Null checks 
SELECT SUM(Company IS NULL) AS null_company, 
	SUM(Product IS NULL) AS null_product, 
    SUM(Price_inr IS NULL) AS null_price 
FROM Laptop_prices;

-- Basic distribution checks 
SELECT TypeName, COUNT(*) AS n, ROUND(AVG(Price_inr),2) AS avg_price 
FROM Laptop_prices 
GROUP BY TypeName 
ORDER BY avg_price DESC;

-- Price band distribution by brand
SELECT 
    Company,
    PriceBand_calc AS PriceBand,
    COUNT(*) AS units,
    ROUND(AVG(Price_inr), 2) AS avg_price
FROM (
    SELECT *,
        CASE 
            WHEN Price_inr < 500 THEN 'Budget'
            WHEN Price_inr < 1000 THEN 'Mid'
            WHEN Price_inr < 1500 THEN 'Upper-Mid'
            WHEN Price_inr < 2500 THEN 'Premium'
            ELSE 'Ultra'
        END AS PriceBand_calc
    FROM Laptop_prices
) t
GROUP BY 
    Company,
    PriceBand_calc
ORDER BY 
    Company,
    PriceBand_calc;

-- Feature bundle performance (SSD vs HDD by RAM bucket)
SELECT
  CASE
    WHEN Ram < 8 THEN '<8'
    WHEN Ram < 16 THEN '8-15'
    WHEN Ram < 32 THEN '16-31'
    ELSE '>=32'
  END AS RamBucket,
  PrimaryStorageType,
  COUNT(*) AS units,
  ROUND(AVG(Price_inr),2) AS avg_price
FROM Laptop_prices
GROUP BY RamBucket, PrimaryStorageType
ORDER BY RamBucket, PrimaryStorageType;

-- Screen resolution impact
SELECT Screen, COUNT(*) AS units, ROUND(AVG(Price_inr),2) AS avg_price
FROM Laptop_prices
GROUP BY Screen
ORDER BY avg_price DESC;
