# ML Dashboard — Student Guide

Welcome! This tool lets you practice the full machine learning workflow — reading in a
dataset, cleaning it, preparing it for modeling, and training a model — while the
dashboard tracks your progress and recommends new assignments as you improve.

## 1. Getting started

1. Open **`ML_Dashboard.ipynb`** with the link to the Google Colab.
2. If the cell does not run because you are missing packages, please install them. 
   (`scikit-learn`, `imbalanced-learn`, `statsmodels`) — this can take a minute.
3. The last cell builds the dashboard and shows the login screen.

## 2. Logging in

- **First time?** Check the terms checkbox and click **Register**. A new 5-digit user ID
  is generated — look for it in the notebook's output area (just below the cell), not in
  the dashboard itself. Write it down, you'll need it every time you come back.
- **Already have an ID?** Type it into the "User ID" field, check the terms box, and
  click **Login**.
- Your progress is tied to this ID. There's no password — anyone with your ID can see
  your data, so don't share it.

## 3. Choosing an assignment

After logging in you'll see a list of recommended assignments, picked based on your
current skill level. You can choose between two modes:

- **Guided** — the dashboard walks you through the assignment step by step. Each step
  comes with a set of hints, but they aren't all shown at once: click the
  **"Instructions"** button to reveal the next hint for the current step. Stuck? Keep
  clicking it — each click adds one more hint, starting vague and getting more specific.
  (The separate **"More info"** button is different: it jumps to the **Topic
  information** tab for background on the topic itself, not a hint for this step.)
- **Monitored** — you work independently; the dashboard compares your end result to a
  reference solution and scores you on it.

The dashboard recommends three assignments at a time, based on your skill level and how
fast you've been improving. Assignments you've already completed with a high score (above
95%) are filtered out.

## 4. The tabs

| Tab | What you do there |
|---|---|
| **Data Information** | Pick a dataset and read it in. Shows a preview, a statistical summary, and missing-value counts. |
| **Data Cleaning** | Deal with missing values (fill with mean/median/mode, or drop rows) and remove irrelevant columns. |
| **Data Processing** | Assign your target column, encode categorical columns, scale features, handle outliers and class imbalance, apply PCA, and split into train/test. |
| **Predictive Modeling** | Train a model (Decision Tree, KNN, Random Forest, Naive Bayes, SGD, or regression) and check its performance (accuracy, confusion matrix, ROC, MSE, R², ...). |
| **Logging** | A running list of every action you've taken. |
| **Topic information** | Background explanation for whatever topic you're currently working on. |

## 5. Made a mistake? You can undo it

Next to the assignment panel there's a button to undo an action:

- **Guided mode:** the button says **"Undo last action"** — it removes only your most
  recent step and puts that step back to "to do".
- **Monitored mode:** use the slider above the button to pick how many of your past
  actions should stay, then click **"Revert to this action"**. Everything after that
  point is undone.

Undoing replays your remaining actions from scratch, so it can take a moment, especially
on larger datasets.

## 6. Picking up where you left off

Logging in again does **not** automatically reopen your last unfinished assignment — you
land back on the assignment-selection screen. Your overall skill progress is remembered,
but a specific in-progress assignment is not resumed automatically.

To continue exactly where you were:

1. Go to the **Learning Path** tab.
2. Pick your previous session from the **performance dropdown**.
3. Click **"Load performance"**. This replays everything you did in that session.

Every time you train a model, your full action history up to that point is saved
automatically — so you'll usually have a recent save point to go back to even if you
closed the notebook without finishing.

## 7. Tips

- Work through the tabs roughly left to right: Data Information → Data Cleaning → Data
  Processing → Predictive Modeling. Later steps (like training a model) often need
  earlier ones (like assigning a target column) to be done first.
- Check the **Logging** tab if you're not sure what you've already done.
- If something looks visually broken (overlapping text, a chart that's cut off), it's
  worth a page refresh / rerunning the cell before assuming it's a real error — but also
  feel free to report it, these things do get fixed.
