CREATE DATABASE IF NOT EXISTS `Coco-cola`;
USE `Coco-cola`;

CREATE TABLE stock_history (
    Date DATE PRIMARY KEY,
    Open FLOAT, High FLOAT, Low FLOAT,
    Close FLOAT, Volume BIGINT,
    Dividends FLOAT, Stock_Splits INT
);

select * from stock_history;

Select Date, (Open+High+Low+Close)/4 As AvgPrice
from stock_history
Order by Date;

Select Year(Date) as Year, Month(Date) as Month,
Sum(volume) as TotalVolume, Avg(Close) as AvgClose
from stock_history
group by Year, Month
Order by Year, Month;
