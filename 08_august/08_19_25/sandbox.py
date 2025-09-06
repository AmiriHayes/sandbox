import pandas as pd

data = {
    'State': ['NewYork', 'Texas', 'California', 'NewYork', 'Texas'],
    'Sales': [250, 180, 300, 120, 400],
    'Category': ['Furniture', 'Office Supplies', 'Technology', 'Furniture', 'Technology'],
    'Quantity': [3, 5, 2, 4, 1],
    'Date': pd.to_datetime(['2024-01-05', '2024-02-10', '2024-03-15', '2024-04-20', '2024-05-25'])
}

df = pd.DataFrame(data)

# code also available in adjacent ipynb file w/ results

df.head(2)
df.tail(2)
df.info()
df.describe()

df.loc[0, "Sales"]
df.iloc[2,1]

df.groupby("State")["Sales"].sum()
df.groupby("Category")["Sales"].transform("mean")

df.dropna()
df.fillna(0)
df.where(df["Sales"] > 200)

df["Sales"].apply(lambda x: x*1.1)
df["State"].map(lambda x: x.upper())

df["State"].value_counts()
df.nlargest(2, "Sales")

pd.melt(df, id_vars="State", value_vars=["Sales", "Quantity"])
df.pivot_table(index="State", values="Sales", aggfunc="mean")
df_ex = df.copy()
df_ex["CategoryList"] = [["A", "B"], ["X"], ["C", "D", "E"], ["F"], ["Y", "Z"]]
print(df_ex.explode("CategoryList"))

df.query("Sales > 200 and Quantity >=3")
df.assign(Discount=df["Sales"]*0.1)
pd.cut(df["Sales"], bins=3, labels=["Low", "Medium", "High"])

df.sort_values(by="Sales", ascending=False)
df.rename(columns={"Sales": "Revenue"})
df.duplicated()
df.drop_duplicates("State")
df.sample(2)
df.corr(numeric_only=True)

# ignore this (didn't know this was possible, did more pdf coding on 8/25)
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics

# Register font for proper rendering
pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3'))

# Create document
doc = SimpleDocTemplate("attempt.pdf", pagesize=LETTER)
styles = getSampleStyleSheet()
story = []

# Title
story.append(Paragraph("ChatGPT: Madrid Study Abroad To-Do List", styles['Title']))
story.append(Spacer(1, 12))

# Sections and items
sections = {
    "Before Leaving for Madrid": [
        "Check passport validity (must be valid for at least 6 months beyond your stay).",
        "Print and/or save digital copies of passport, visa/student residence permit (if required), flight ticket, housing contract, and insurance.",
        "Confirm you have health insurance coverage abroad (study abroad program usually requires it).",
        "Apply for ISIC (International Student ID Card) if you want discounts.",
        "Order some euros in cash from your bank (at least enough for your first week: ~€200–300).",
        "Call your U.S. bank to notify them you’ll be abroad.",
        "Research whether to open a Spanish bank account (depends if your program requires it).",
        "Unlock your phone so you can use a Spanish SIM or eSIM.",
        "Research Spanish providers (Movistar, Vodafone, Orange, MásMóvil, Simyo, Lowi).",
        "Make a packing list (clothes, toiletries, electronics, adapters—Spain uses Type C/F plugs, 230V).",
        "Check airline baggage limits.",
        "Research UPS/DHL costs for mailing appliances (shipping is expensive—might be cheaper to buy locally).",
        "Buy underwear + any last-minute clothes.",
        "Download Google Translate (offline Spanish).",
        "Download Madrid Metro map + transportation apps (Metro de Madrid, Moovit, Google Maps).",
        "Check with your program about orientation, emergency contacts, and required paperwork.",
        "Back up your computer/phone."
    ],
    "Upon Arrival in Madrid (First Week)": [
        "Buy/activate Spanish SIM or eSIM.",
        "Decide if you need a Spanish bank account or if Wise/Revolut is enough.",
        "Get transportation card: Abono Joven (under 26) is ~€20/month unlimited Metro, bus, Cercanías.",
        "Check in with landlord/flatmates.",
        "Buy basics for your room/apartment (bedding, toiletries, cleaning stuff).",
        "Grocery shop + scope out nearest supermarket (Mercadona, Carrefour, Lidl).",
        "Register at your host university.",
        "Get your student ID.",
        "Attend orientation sessions.",
        "Ask about student discounts (museums, gyms, travel).",
        "Learn where your nearest pharmacy and health clinic are.",
        "Save emergency numbers (112 = Spain’s 911).",
        "Explore your neighborhood on foot.",
        "Locate laundromat or laundry facilities.",
        "Find local spots for food/coffee/study."
    ]
}

# Add sections to story
for section, items in sections.items():
    story.append(Paragraph(section, styles['Heading2']))
    story.append(Spacer(1, 6))
    checklist = ListFlowable(
        [ListItem(Paragraph(item, styles['Normal'])) for item in items],
        bulletType='bullet', start='circle'
    )
    story.append(checklist)
    story.append(Spacer(1, 12))

# Build PDF
doc.build(story)