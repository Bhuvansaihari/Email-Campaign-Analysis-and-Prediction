# Email Campaign Analysis and Prediction

This project analyzes an email marketing campaign, identifies user behavior patterns, and builds a machine learning model to predict users who are likely to click on email links. 

## 📁 Files Used

- `email_table.csv`: Basic user and email information.
- `email_opened_table.csv`: Records of emails that were opened.
- `link_clicked_table.csv`: Records of users who clicked on links.

## 🔍 Objectives

1. **Analyze** open and click rates.
2. **Build** a predictive model using Random Forest to identify potential clickers.
3. **Simulate** a targeted campaign to improve Click-Through Rate (CTR).
4. **Identify** interesting patterns based on user and email attributes.

## 🛠️ Libraries Used

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

## 💡 How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt

## 📥 Download Dataset

Please download the required CSV files from the following Google Drive links:

- [email_table.csv](https://drive.google.com/file/d/1e34MkRjtu9lA1dny34ZD_VwXE-Fb4hDk/view?usp=drive_link)
- [email_opened_table.csv](https://drive.google.com/file/d/16Xz_WUa-NSVvKX__3wsUW0zq3pDa0lg0/view?usp=drive_link)
- [link_clicked_table.csv](https://drive.google.com/file/d/1oWmwhHfeW93WVuvmMnpgA3CSsf-PfP0k/view?usp=drive_link)

> ⚠️ After downloading, **update the file paths** in the script to reflect the location where you've saved the files on your system. For example:

```python
email_df = pd.read_csv(r"path/to/email_table.csv")
opened_df = pd.read_csv(r"path/to/email_opened_table.csv")
clicked_df = pd.read_csv(r"path/to/link_clicked_table.csv")
