from reportlab.platypus import Paragraph, Spacer, PageBreak
from styles import styles, p, bullet, num_bullet, callout, divider

def get_chapter1_story():
    story = []
    story.append(Paragraph("Chapter 1: The Intuitive Foundation — Credit Risk for a 12-Year-Old", styles['ChapterTitle']))
    story.append(Paragraph("Understanding the Human, Financial, and Physical Reality of Lending, Default, and Systematic Shocks", styles['CoverSubtitle']))
    story.append(divider())

    story.append(p(
        "Before diving into differential equations, logistic loss gradients, and covariance matrices, "
        "it is essential to understand the real-world story behind this project. Every mathematical "
        "formula in credit risk is merely a formal language describing human promises, financial survival, "
        "and unexpected economic storms."
    ))

    story.append(Paragraph("1.1 What Is a Loan and What Does 'Default' Mean?", styles['SectionTitle']))
    story.append(p(
        "Imagine you have saved $100 from your birthday money. Your friend Alex comes to you and says: "
        "<i>'I want to start a weekend lemonade stand, but I need $100 to buy lemons, sugar, cups, and a wooden table. "
        "If you lend me $100 today, I promise to pay you back $10 every single month for the next 12 months.'</i>"
    ))
    story.append(p(
        "If Alex keeps his promise, you will receive $120 total ($10 × 12). The extra $20 is the <b>interest</b>—your reward "
        "for trusting Alex and waiting a whole year to get your money back. In financial language, you are the <b>Lender</b>, "
        "Alex is the <b>Borrower</b>, the $100 is the <b>Principal (Loan Amount)</b>, and the agreement is an <b>Installment Loan</b>."
    ))
    story.append(p(
        "Now, what happens if after paying you back for 4 months ($40), Alex suddenly stops paying completely? "
        "Maybe his stand broke, or he lost interest, or he spent his money elsewhere. He tells you: <i>'I have no money left, and I cannot pay you the remaining $60.'</i> "
        "That event is called a <b>Default</b>. In banking, when the lender accepts that the money will never be collected, "
        "they write it off as a <b>Charge-Off</b>. You just lost $60 of your hard-earned money."
    ))

    story.append(callout(
        "Core Definition: Probability of Default (PD)",
        [
            "In retail banking, a borrower is not simply labeled 'good' or 'bad'. Nobody knows the future with 100% certainty. "
            "Instead, banks calculate a number between 0% and 100% called the <b>Probability of Default (PD)</b>. "
            "A PD of 3% means: out of 100 people just like Alex, we expect 3 of them will fail to repay. "
            "A PD of 25% means: 1 out of every 4 will go broke."
        ],
        kind='tip'
    ))

    story.append(Paragraph("1.2 The Three Personal Clues (Borrower Micro-Data)", styles['SectionTitle']))
    story.append(p(
        "Before you handed Alex your $100, you looked at three pieces of information to decide if he was trustworthy:"
    ))
    story.append(num_bullet("1", "<b>His Past Track Record (FICO Credit Score):</b> Has Alex borrowed money or toys from friends before? Did he always return them on time, or did he forget and make excuses? In America, a company called Fair Isaac Corporation tracks every adult's borrowing history and boils it down to a single score between <b>300 and 850</b> called the <b>FICO Score</b>. If Alex has a 780 FICO score, he has never missed a payment in his life. If his score is 610, he has a messy track record."))
    story.append(num_bullet("2", "<b>His Existing Debt Burden (Debt-to-Income / DTI):</b> Alex gets a $50 monthly allowance from his parents (his <b>Income</b>). But you find out he already promised to pay $25 each month to his older brother for a bicycle! That means <b>50%</b> of his income is already spoken for! In banking, this is called the <b>Debt-to-Income (DTI) Ratio</b>. If your DTI is too high, even a tiny emergency will leave you with zero cash to pay your debts."))
    story.append(num_bullet("3", "<b>His Total Cash Capacity (Annual Income):</b> If Alex makes $500 a month instead of $50, an unexpected $10 expense is no big deal. The higher your stable income, the easier it is to absorb life's bumps."))

    story.append(p(
        "These three numbers represent <b>Microeconomic Borrower Fundamentals</b>. For 50 years, banks built their entire business "
        "almost exclusively on these micro-clues. If a borrower had a 750 FICO, a 15% DTI, and a $90,000 salary, they were approved instantly."
    ))

    story.append(Paragraph("1.3 The Unexpected Storm (Macroeconomic Shocks)", styles['SectionTitle']))
    story.append(p(
        "Now comes the heart of this research project. Imagine that Alex is the most honest, hardworking kid in school. "
        "His FICO score is pristine. He sets up his lemonade stand on June 1st. But in July, a freak <b>economic storm</b> hits the town:"
    ))
    story.append(bullet("<b>The Factory Closes (Unemployment Spikes - UNRATE):</b> The big car factory in town shuts down, and 15% of the parents lose their jobs. Nobody has money to buy lemonade anymore."))
    story.append(bullet("<b>Lemon Prices Skyrocket (Inflation - CPI):</b> A disease destroys lemon crops across the country. The price of lemons jumps from $0.50 to $2.50. Alex's costs explode, eating up all his profit margin."))
    story.append(bullet("<b>The Speed of the Crash (Shock Velocity - UNRATE Change):</b> It didn't just happen slowly over 10 years; the factory closed in <i>two weeks</i>! Alex had zero time to adapt, find another job, or sell a different drink."))

    story.append(p(
        "Under this ferocious storm, Alex defaults. Not because he lied, not because he was lazy, but because the <b>Systematic Macroeconomic Environment</b> "
        "crushed his ability to earn money. This is the difference between <b>Idiosyncratic Risk</b> (Alex's personal habits) "
        "and <b>Systematic Risk</b> (the economy-wide storm that affects everyone at once)."
    ))

    story.append(Paragraph("1.4 The Central Research Question: The Origination Weather Dilemma", styles['SectionTitle']))
    story.append(p(
        "This project investigates a very specific and controversial question in modern quantitative risk:"
    ))
    story.append(callout(
        "The Million-Dollar Question",
        [
            "<i>'When Alex walks into the bank on June 1st to ask for a 36-month loan, does looking at the national economic weather report on that exact day help the bank predict whether Alex will default over the next 3 years—better than just looking at Alex's personal FICO score, income, and debt?'</i>"
        ],
        kind='amber'
    ))
    story.append(p(
        "Think about why this is tricky! On June 1st, 2012, the national unemployment rate was 8.2%. "
        "Does that 8.2% tell you who will fail in 2014? What if by 2014 the economy booms and unemployment drops to 5.5%? "
        "Conversely, in 2007, unemployment was a peaceful 4.5%. Looking at the sunny weather in 2007 gave you zero warning that the Great Financial Crisis of 2008 was about to wipe out the banking system!"
    ))
    story.append(p(
        "Our mission in this project is to take <b>2.26 million real loans</b> from LendingClub, combine them with official Federal Reserve economic data, "
        "and use rigorous mathematical models to find out once and for all: <b>Does macroeconomic data at the time of loan signing actually improve default prediction?</b>"
    ))
    story.append(PageBreak())
    return story
