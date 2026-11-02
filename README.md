### AUTOMOBILE
The objective of this analysis is to apply exploratory data analysis and statistical techniques to the automobile dataset in order to identify the key variables that influence car price.
The dataset contained  201 entries  about cars and 29 columns  with car specifications and costs. 19 columns had continuous variable and the rest were categorical.
## DATA CLEANING
- The stroke column had 4 missing values. The average stroke value per car type was imputed .The row with the missing value in the horse-power binned column was deleted .
- There were no duplicates in the dataset.
- On outliers, none was deleted because the variations were associated with the varying makes of the cars, for example certain cars like the Peugeot had significantly higher compression ratios than the Audi.
- The cleaned dataset with 200 entries was re named ``melody.csv’’
## DESCRIPTIVE STATISTICS
Continuous variables were summarized using means, medians and modes. Frequency counts for categorical variables were calculated.
## EFFECT OF INDEPENDENT VARIABLES ON PRICE.
The heatmap with continuous variables showed that horse power, engine-size and curb weight have a strong positive(>+0.8) correlation with price. City-l/100km and width had a fairly strong positive (>+0.7<+0.8) correlation with price. City.mpg and highway.mpg had  moderate negative correlation with price at -0.68 and -0.7 respectively.
For categorical variables, the mean price per category was calculated with the highest and lowest subcategory as follows:
- Drive wheels( rwd,fwd)
- Body style (hardtop, hatchback)
- Make ( jaguar, Chevrolet)
- Aspiration( turbo, std)
- Num of doors (four, two)
- Engine location(rear, front)
- Engine type (ohcv, ohc)
- Num of cylinders(eight, three)
- Fuel system(mpfi, 2bl)
- Horse-power-binned (high, low)

Using recursive feature elimination, curb-weight, horse-power and engine-size are the strongest predictors of car price.
## RECOMMENDATION
Car dealers should consider car specifications like engine type and size when deciding of a car price. Also consider the possible longevity of the car considering the road type in a specific area so that end users get optimum benefits from their cars.
