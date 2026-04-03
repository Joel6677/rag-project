import pandas as pd


def load_csv():
    df = pd.read_csv('Sample - Superstore.csv', encoding='latin-1')
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter

    return df


def create_row_documents(df):
    documents = []

    for _, row in df.iterrows():
        text = (
            f"Order {row['Order ID']} on {row['Order Date'].strftime('%Y-%m-%d')}: "
            f"Customer {row['Customer Name']} in {row['City']}, {row['State']} ({row['Region']} region) "
            f"bought {row['Quantity']} unit(s) of '{row['Product Name']}' "
            f"(Category: {row['Category']}, Sub-Category: {row['Sub-Category']}) "
            f"for ${row['Sales']:.2f} with a discount of {row['Discount']*100:.0f}%. "
            f"Profit was ${row['Profit']:.2f}."
        )
        metadata = {
            "chunk_type": "row",
            "region": row["Region"],
            "category": row["Category"],
            "sub_category": row["Sub-Category"],
            "year": int(row["Year"]),
            "quarter": int(row["Quarter"]),
            "month": int(row["Month"]),
        }
        documents.append({"text": text, "metadata": metadata})

    return documents


def create_aggregated_documents(df):
    documents = []

    # Yearly summaries
    for year, group in df.groupby("Year"):
        text = (
            f"Year {year} summary: Total sales were ${group['Sales'].sum():,.2f}, "
            f"total profit was ${group['Profit'].sum():,.2f}, "
            f"with {len(group)} orders across all regions and categories."
        )
        documents.append({"text": text, "metadata": {
                         "chunk_type": "yearly_summary", "year": int(year)}})
    # Regional summaries
    for region, group in df.groupby("Region"):
        text = (
            f"{region} region summary: Total sales ${group['Sales'].sum():,.2f}, "
            f"profit ${group['Profit'].sum():,.2f}, "
            f"{len(group)} orders, average discount {group['Discount'].mean()*100:.1f}%."
        )
        documents.append({"text": text, "metadata": {
                         "chunk_type": "regional_summary", "region": region}})

    # Category summaries
    for category, group in df.groupby("Category"):
        text = (
            f"{category} category summary: Total sales ${group['Sales'].sum():,.2f}, "
            f"profit ${group['Profit'].sum():,.2f}, "
            f"{len(group)} orders, average discount {group['Discount'].mean()*100:.1f}%."
        )
        documents.append({"text": text, "metadata": {
                         "chunk_type": "category_summary", "category": category}})

    # Region + Year summaries
    for (region, year), group in df.groupby(["Region", "Year"]):
        text = (
            f"{region} region in {year}: Sales ${group['Sales'].sum():,.2f}, "
            f"profit ${group['Profit'].sum():,.2f}, {len(group)} orders."
        )
        documents.append({"text": text, "metadata": {
                         "chunk_type": "region_year_summary", "region": region, "year": int(year)}})

    return documents


def create_statistical_documents(df):
    documents = []

    # Top performing sub-categories
    top_subs = df.groupby(
        "Sub-Category")["Profit"].sum().sort_values(ascending=False)
    top_text = "Top 5 sub-categories by profit: " + ", ".join(
        [f"{sub} (${profit:,.2f})" for sub, profit in top_subs.head(5).items()]
    )
    documents.append({"text": top_text, "metadata": {
                     "chunk_type": "statistical_summary"}})

    # Bottom performing sub-categories
    bottom_text = "Bottom 5 sub-categories by profit: " + ", ".join(
        [f"{sub} (${profit:,.2f})" for sub, profit in top_subs.tail(5).items()]
    )
    documents.append({"text": bottom_text, "metadata": {
                     "chunk_type": "statistical_summary"}})

    # Profit margin by category
    for category, group in df.groupby("Category"):
        margin = (group["Profit"].sum() / group["Sales"].sum()) * 100
        text = (
            f"{category} profit margin: {margin:.1f}%. "
            f"Mean sale value ${group['Sales'].mean():,.2f}, "
            f"median profit ${group['Profit'].median():,.2f}."
        )
        documents.append({"text": text, "metadata": {
                         "chunk_type": "statistical_summary", "category": category}})

    # Most discounted sub-categories
    top_discount = df.groupby(
        "Sub-Category")["Discount"].mean().sort_values(ascending=False)
    discount_text = "Most discounted sub-categories: " + ", ".join(
        [f"{sub} ({disc*100:.1f}%)" for sub,
         disc in top_discount.head(5).items()]
    )
    documents.append({"text": discount_text, "metadata": {
                     "chunk_type": "statistical_summary"}})

    return documents


def create_text_documents(df):
    row_docs = create_row_documents(df)
    agg_docs = create_aggregated_documents(df)
    stat_docs = create_statistical_documents(df)
    all_docs = row_docs + agg_docs + stat_docs
    return all_docs


def main():
    df = load_csv()
    all_docs = create_text_documents(df)
    print(all_docs[0])


if __name__ == "__main__":
    main()
