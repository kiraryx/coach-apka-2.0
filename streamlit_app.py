import streamlit as st
st.set_page_config(page_title="Credit Coach BCC", page_icon="🏦")
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib

# Кастомизация темы
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;700&display=swap');

    * {
        font-family: 'Montserrat', sans-serif !important;
    }
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif !important;
    }
    button, input, select, label, .stButton, .stTextInput, .stSlider, .stSelectbox {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 500 !important;
        font-size: 18px !important;
    }
    .main {
        background-color: #FFFFFF !important;
    }
    .stButton>button {
        background-color: #00A859;
        color: white;
        border-radius: 5px;
        display: block;
        margin: 0 auto;
        padding: 10px 20px;
    }
    .stMetric {
        border: 2px solid #00A859;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
        margin: 0 auto;
        width: fit-content;
    }
    .stMetric.red {
        border-color: #FF0000 !important;
        color: #FF0000 !important;
    }
    h1, h2, h3 {
        color: #00A859;
        text-align: center !important;
    }
    .center-text {
        text-align: center !important;
        color: #333;
    }
    .approval-circle {
        width: 300px;
        height: 300px;
        border-radius: 50%;
        border: 3px solid #00A859;
        background-color: transparent;
        color: #00A859;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 36px;
        font-weight: bold;
        margin: 20px auto;
    }
    .approval-circle.red {
        border-color: #FF0000 !important;
        color: #FF0000 !important;
    }
    .approval-text {
        text-align: center;
        color: #00A859;
        font-weight: bold;
    }
    .approval-text.red {
        color: #FF0000 !important;
    }
    .stSlider, .stTextInput, .stSelectbox, .stForm {
        background-color: #FFFFFF !important;
    }
    .stSlider > div, .stTextInput > div, .stSelectbox > div {
        background-color: #FFFFFF !important;
    }
    /* Стили для чат-бота */
    .chatbot-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 60px;
        height: 60px;
        background-color: #00A859;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        cursor: pointer;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        z-index: 1000;
    }
    .chatbot {
        position: fixed;
        bottom: 90px;
        right: 20px;
        width: 350px;
        height: 500px;
        background-color: #f9f9f9;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        display: flex;
        flex-direction: column;
        z-index: 1000;
    }
    .chatbot-header {
        background-color: #00A859;
        color: white;
        padding: 10px;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .chatbot-body {
        flex: 1;
        padding: 10px;
        overflow-y: auto;
        background-color: #fff;
    }
    .chatbot-input {
        display: flex;
        padding: 10px;
        border-top: 1px solid #ddd;
    }
    .chatbot-input input {
        flex: 1;
        padding: 8px;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-right: 10px;
    }
    .chatbot-input button {
        padding: 8px 15px;
        background-color: #00A859;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .chatbot-message {
        margin: 5px 0;
        padding: 8px;
        border-radius: 5px;
    }
    .chatbot-message.user {
        background-color: #00A859;
        color: white;
        text-align: right;
    }
    .chatbot-message.bot {
        background-color: #e0e0e0;
        text-align: left;
    }
    .logo-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Добавление изображений
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
st.image("images/bcclog.png", width=600, use_container_width=False)
st.image("images/aichatlogo.png", width=200, use_column_width="always")
st.markdown('</div>', unsafe_allow_html=True)

# Добавим стиль для центрирования
st.markdown("""
    <style>
    .logo-container img {
        display: block !important;
        margin: 0 auto !important;
    }
    </style>
""", unsafe_allow_html=True)

# Генерация синтетической базы данных
np.random.seed(42)
n_samples = 1000
data = {
    'income': np.random.normal(300000, 50000, n_samples),
    'late_payments': np.random.randint(0, 5, n_samples),
    'account_balance': np.random.normal(50000, 10000, n_samples),
    'fines': np.random.randint(0, 3, n_samples),
    'tax_debt': np.random.normal(5000, 2000, n_samples),
    'active_loans': np.random.randint(0, 5, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples)
}
df = pd.DataFrame(data)
X = df[['income', 'late_payments', 'account_balance', 'fines', 'tax_debt', 'active_loans']]
y = df['credit_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Функция-заглушка для API eGov.kz
def check_egov_data(iin):
    if not iin or len(iin) != 12 or not iin.isdigit():
        return 0, 0.0
    hash_val = int(hashlib.sha256(iin.encode()).hexdigest(), 16)
    fines = (hash_val % 3) + (1 if (hash_val % 100) > 80 else 0)
    tax_debt = abs(np.random.normal(5000, 2000) + (hash_val % 1000))
    return fines, round(tax_debt, 2)

# Функция для оценки вероятности одобрения (беззалоговый кредит)
def approval_probability(credit_score, monthly_payment, income, active_loans, loan_payments, late_payments, fines, tax_debt, age, loan_amount, loan_term_months):
    free_balance = income - loan_payments
    max_payment = free_balance * 0.4  # 40% свободного баланса

    # Проверка условий БЦК
    if loan_amount < 60000 or loan_amount > 7_000_000:
        return False, f"Сумма кредита должна быть от 60,000 ₸ до 7,000,000 ₸ (запрошено: {loan_amount:,} ₸)", 0
    if loan_term_months < 6 or loan_term_months > 60:
        return False, f"Срок кредита должен быть от 6 до 60 месяцев (запрошено: {loan_term_months} месяцев)", 0

    # Базовая вероятность на основе кредитного рейтинга
    probability = 0
    if credit_score >= 750:
        probability = 90
    elif credit_score >= 650:
        probability = 75
    elif credit_score >= 600:
        probability = 50
    elif credit_score >= 500:
        probability = 30
    else:
        probability = 10

    # Уменьшение за превышение долговой нагрузки (с учётом процентной ставки 24,43% ГЭСВ)
    annual_rate = 0.2443  # Средняя ГЭСВ
    monthly_rate = annual_rate / 12
    adjusted_monthly_payment = (monthly_payment * monthly_rate * (1 + monthly_rate) ** loan_term_months) / ((1 + monthly_rate) ** loan_term_months - 1)
    if adjusted_monthly_payment > max_payment:
        excess_ratio = (adjusted_monthly_payment - max_payment) / max_payment
        probability -= min(40, excess_ratio * 40)  # Снижение до 40%

    # Уменьшение за просрочки
    if late_payments > 0:
        probability -= late_payments * 15  # За каждую просрочку -15%

    # Уменьшение за штрафы
    if fines > 0:
        probability -= fines * 10  # За каждый штраф -10%

    # Уменьшение за налоговую задолженность
    if tax_debt > 0:
        probability -= min(25, tax_debt / 5000 * 15)  # До 25% снижения за налоги

    # Уменьшение за активные кредиты
    if active_loans > 2:
        probability -= (active_loans - 2) * 10  # За каждый лишний кредит -10%

    # Уменьшение за возраст
    if age > 68:
        probability -= 30  # Пенсионерам старше 68 лет сложнее получить кредит

    probability = max(0, min(100, probability))
    approved = probability >= 60
    return approved, f"Вероятность одобрения: {probability:.0f}%", probability

# Функция для расчёта залогового кредита
def collateral_loan_calculation(collateral_value, loan_amount, loan_term_months, income, loan_payments, age, iin):
    fines, tax_debt = check_egov_data(iin)

    # Проверка условий БЦК
    max_loan_amount = collateral_value * 0.7  # До 70% от стоимости залога
    if loan_amount > max_loan_amount:
        return 0, 0, False, 0, f"Сумма кредита превышает 70% от стоимости залога (максимум: {max_loan_amount:,} ₸)"
    if loan_amount > 150_000_000:
        return 0, 0, False, 0, "Сумма кредита превышает максимум (150 млн ₸)"
    if loan_term_months < 3 or loan_term_months > 120:
        return 0, 0, False, 0, f"Срок кредита должен быть от 3 до 120 месяцев (запрошено: {loan_term_months} месяцев)"

    # Процентная ставка 22,5% годовых (средняя)
    annual_rate = 0.225
    monthly_rate = annual_rate / 12
    monthly_payment = (loan_amount * monthly_rate * (1 + monthly_rate) ** loan_term_months) / ((1 + monthly_rate) ** loan_term_months - 1)

    # Вероятность одобрения
    free_balance = income - loan_payments
    max_payment = free_balance * 0.4
    probability = 70  # Базовая вероятность выше из-за залога

    if monthly_payment > max_payment:
        excess_ratio = (monthly_payment - max_payment) / max_payment
        probability -= min(40, excess_ratio * 40)

    if age > 68:
        probability -= 30

    if fines > 0:
        probability -= fines * 10

    if tax_debt > 0:
        probability -= min(25, tax_debt / 5000 * 15)

    probability = max(0, min(100, probability))
    approved = probability >= 60
    return max_loan_amount, monthly_payment, approved, probability, ""

# Функция для бизнес-кредита
def business_loan_calculation(loan_amount, loan_term_months, income, loan_payments, business_type, purpose, iin):
    fines, tax_debt = check_egov_data(iin)

    # Проверка условий БЦК
    max_loan = 20_000_000 if business_type == "ИП" else 100_000_000
    max_term = 36 if purpose == "Пополнение оборотных средств" else 120
    if loan_amount < 150_000 or loan_amount > max_loan:
        return 0, False, 0, f"Сумма кредита должна быть от 150,000 ₸ до {max_loan:,} ₸ (запрошено: {loan_amount:,} ₸)"
    if loan_term_months < 6 or loan_term_months > max_term:
        return 0, False, 0, f"Срок кредита должен быть от 6 до {max_term} месяцев (запрошено: {loan_term_months} месяцев)"

    # Процентная ставка 24,45% (средняя для беззалогового)
    annual_rate = 0.2445
    monthly_rate = annual_rate / 12
    monthly_payment = (loan_amount * monthly_rate * (1 + monthly_rate) ** loan_term_months) / ((1 + monthly_rate) ** loan_term_months - 1)

    # Вероятность одобрения
    free_balance = income - loan_payments
    max_payment = free_balance * 0.4
    probability = 60  # Базовая вероятность

    if monthly_payment > max_payment:
        excess_ratio = (monthly_payment - max_payment) / max_payment
        probability -= min(40, excess_ratio * 40)

    if fines > 0:
        probability -= fines * 10

    if tax_debt > 0:
        probability -= min(25, tax_debt / 5000 * 15)

    probability = max(0, min(100, probability))
    approved = probability >= 60
    return monthly_payment, approved, probability, ""

# Функция рекомендаций (для беззалогового кредита)
def generate_recommendations(client_data, loan_amount, loan_term_months, iin, loan_payments, age):
    fines, tax_debt = check_egov_data(iin)
    client_data[3] = fines
    client_data[4] = tax_debt
    current_score = model.predict([client_data])[0]
    monthly_payment = loan_amount / loan_term_months
    free_balance = client_data[0] - loan_payments
    recommendations = []
    loan_suggestions = []
    quick_actions = []

    if client_data[1] > 0:
        rec_text = (
            f"Уважаемый клиент, у вас есть {client_data[1]} просроченных платежей. "
            "Просрочки могут существенно снижать ваш кредитный рейтинг. "
            "Рекомендуем погасить их как можно скорее. Для этого вы можете воспользоваться приложением БЦК или посетить ближайшее отделение."
        )
        recommendations.append(rec_text)
        quick_actions.append(("Погасить просроченные платежи", "https://cabinet.bcc.kz"))

    if client_data[3] > 0:
        rec_text = (
            f"Мы заметили, что у вас есть {client_data[3]} неоплаченных штрафов, возможно, за нарушение ПДД. "
            "Неоплаченные штрафы могут негативно влиять на решение по кредиту. "
            "Советуем оплатить их через портал eGov.kz. После оплаты данные обновятся в течение 3–5 рабочих дней."
        )
        recommendations.append(rec_text)
        quick_actions.append(("Оплатить штрафы за ПДД", "https://egov.kz"))

    if client_data[4] > 0:
        rec_text = (
            f"Уважаемый клиент, у вас обнаружена налоговая задолженность в размере {client_data[4]:.2f} ₸. "
            "Это может стать причиной отказа в выдаче кредита. "
            "Рекомендуем погасить задолженность через портал eGov.kz в разделе 'Налоги'. "
            "После оплаты данные обновятся в течение 5–7 рабочих дней, что улучшит ваши шансы на одобрение."
        )
        recommendations.append(rec_text)
        quick_actions.append(("Погасить налоговую задолженность", "https://egov.kz/services/10108"))

    if client_data[5] > 2:
        rec_text = (
            f"У вас сейчас {client_data[5]} активных кредитов, что создаёт высокую долговую нагрузку. "
            "Для улучшения кредитного рейтинга советуем закрыть хотя бы один из кредитов. "
            "Вы можете сделать это через приложение БЦК или в отделении банка."
        )
        recommendations.append(rec_text)
        quick_actions.append(("Закрыть один из кредитов", "https://cabinet.bcc.kz/loans"))

    if free_balance < monthly_payment * 0.4:
        rec_text = (
            f"Ваш свободный баланс после текущих платежей составляет {free_balance:.2f} ₸, "
            f"а ежемесячный платёж по желаемому кредиту — {monthly_payment:.2f} ₸. "
            "Это превышает допустимую нагрузку (40% от свободного баланса). "
            "Рекомендуем рассмотреть либо уменьшение суммы кредита, либо увеличение срока кредитования."
        )
        recommendations.append(rec_text)

    approved, approval_reason, probability = approval_probability(
        current_score, monthly_payment, client_data[0], client_data[5], loan_payments, client_data[1], fines, tax_debt, age, loan_amount, loan_term_months
    )

    if not approved:
        new_term = int(loan_amount / (free_balance * 0.4)) + 1 if free_balance > 0 else loan_term_months
        new_payment = loan_amount / new_term
        loan_suggestions.append(
            f"Попробуйте увеличить срок кредита до {new_term} месяцев — это снизит ежемесячный платёж до {new_payment:.2f} ₸, "
            "что может повысить вероятность одобрения."
        )
        new_amount = free_balance * 0.4 * loan_term_months
        loan_suggestions.append(
            f"Или уменьшите сумму кредита до {new_amount:.2f} ₸ при текущем сроке ({loan_term_months} месяцев), "
            "чтобы платёж стал более комфортным для вашего бюджета."
        )
        quick_actions.append(("Подобрать другой кредит", "https://bcc.kz/loans"))

    if not recommendations:
        recommendations.append(
            "Уважаемый клиент, ваши финансовые показатели в отличном состоянии! "
            "Продолжайте следить за своим рейтингом через приложение БЦК."
        )

    if probability >= 60:
        quick_actions.append(("Оформить кредит онлайн", "https://cabinet.bcc.kz/loans/apply"))

    return recommendations, loan_suggestions, current_score, approved, probability, quick_actions

# Функция для анализа факторов клиента
def analyze_client_factors(client_data, importance):
    factors = ['Доходы', 'Просрочки', 'Штрафы', 'Налоги', 'Кредиты']
    client_factors = []

    if client_data[0] > 300000:
        client_factors.append(f"Ваши доходы ({client_data[0]:.0f} ₸) выше среднего, это положительно влияет на рейтинг.")
    else:
        client_factors.append(f"Ваши доходы ({client_data[0]:.0f} ₸) ниже среднего, это может снижать рейтинг.")

    if client_data[1] > 0:
        client_factors.append(f"У вас есть {client_data[1]} просроченных платежей, это негативно влияет на рейтинг.")
    else:
        client_factors.append("У вас нет просрочек, это положительно влияет на рейтинг.")

    if client_data[3] > 0:
        client_factors.append(f"У вас есть {client_data[3]} неоплаченных штрафов, это снижает ваш рейтинг.")
    else:
        client_factors.append("У вас нет неоплаченных штрафов, это положительно влияет на рейтинг.")

    if client_data[4] > 0:
        client_factors.append(f"Налоговая задолженность ({client_data[4]:.2f} ₸) негативно влияет на ваш рейтинг.")
    else:
        client_factors.append("У вас нет налоговой задолженности, это положительно влияет на рейтинг.")

    if client_data[5] > 2:
        client_factors.append(f"У вас {client_data[5]} активных кредитов, высокая долговая нагрузка снижает рейтинг.")
    else:
        client_factors.append(f"У вас {client_data[5]} активных кредитов, это допустимая нагрузка для вашего рейтинга.")

    return client_factors

# Навигация
st.sidebar.title("Навигация")
page = st.sidebar.selectbox("Выберите страницу", ["Беззалоговый кредит", "Залоговый кредит", "Кредиты для бизнеса"])

# Сохранение данных клиента в сессии
if "client_data" not in st.session_state:
    st.session_state.client_data = {
        "income": 400000,
        "loan_payments": 20000
    }

# Первая страница: Беззалоговый кредит
if page == "Беззалоговый кредит":
    st.markdown("<h1 style='font-weight: 700;'>Credit Coach BCC</h1>", unsafe_allow_html=True)
    st.markdown("""
    <h3 class='center-text'>
    Проверьте ваш кредитный рейтинг и получите персональные рекомендации
    </h3>
    """, unsafe_allow_html=True)

    with st.form("client_form"):
        st.markdown("**Ваши данные**")
        iin = st.text_input("ИИН (12 цифр, обязательно)", max_chars=12)
        if not iin or len(iin) != 12 or not iin.isdigit():
            st.error("Пожалуйста, введите корректный ИИН (12 цифр).")

        cols = st.columns(2)
        with cols[0]:
            income = st.number_input("Официальная заработная плата (₸)", min_value=0, value=st.session_state.client_data["income"], step=1000)
            late_payments = st.selectbox("Просрочки", [0, 1, 2, 3, 4, 5])
            age = st.number_input("Ваш возраст", min_value=18, value=30, step=1)
        with cols[1]:
            loan_payments = st.number_input("Платежи по текущим кредитам (₸/месяц)", min_value=0, value=st.session_state.client_data["loan_payments"], step=1000)
            active_loans = st.selectbox("Активные кредиты", [0, 1, 2, 3, 4, 5])

        st.markdown("<div style='margin-top: 20px;'><b>Параметры желаемого кредита</b></div>", unsafe_allow_html=True)
        cols = st.columns(2)
        with cols[0]:
            loan_amount = st.number_input("Сумма кредита (₸)", min_value=60000, max_value=7000000, value=1000000, step=10000)
        with cols[1]:
            loan_term_months = st.slider("Срок кредита (месяцы)", 6, 60, 12)

        submitted = st.form_submit_button("Проверить")
        if submitted and (not iin or len(iin) != 12 or not iin.isdigit()):
            st.error("Пожалуйста, заполните поле ИИН корректно.")
            submitted = False

    if submitted:
        st.session_state.client_data["income"] = income
        st.session_state.client_data["loan_payments"] = loan_payments

        client_data = [income, late_payments, 50000, 0, 0, active_loans]
        recommendations, loan_suggestions, current_score, approved, probability, quick_actions = generate_recommendations(
            client_data, loan_amount, loan_term_months, iin, loan_payments, age
        )

        if approved:
            st.markdown("**Кредит может быть одобрен ✅**", unsafe_allow_html=True)
        else:
            st.markdown("**Кредит не одобрен ❌**", unsafe_allow_html=True)

        st.header("Результаты")
        color_class = "red" if probability < 60 else ""
        st.markdown(f"""
            <div class="approval-circle {color_class}">{probability:.0f}%</div>
            <p class="approval-text {color_class}">Вероятность одобрения</p>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <style>
            .stMetric {{
                border-color: {'#FF0000' if probability < 60 else '#00A859'} !important;
                color: {'#FF0000' if probability < 60 else '#00A859'} !important;
            }}
            </style>
        """, unsafe_allow_html=True)
        st.metric("Кредитный рейтинг", f"{current_score:.0f}")

        button_color = "#FF0000" if probability < 60 else "#00A859"
        st.markdown(f"""
            <style>
            .stButton>button {{
                background-color: {button_color} !important;
                transition: background-color 0.3s;
            }}
            </style>
        """, unsafe_allow_html=True)
        for action_text, url in quick_actions[-1:]:
            st.link_button(action_text, url)

        st.subheader("Персональные рекомендации")
        for rec in recommendations:
            st.write(f"- {rec}")

        if loan_suggestions:
            st.subheader("Оптимизация кредита")
            for suggestion in loan_suggestions:
                st.write(f"- {suggestion}")

        st.subheader("Факторы влияния на ваш рейтинг")
        client_factors = analyze_client_factors(client_data, model.feature_importances_)
        for factor in client_factors:
            st.write(f"- {factor}")

        factors = ['Доходы', 'Просрочки', 'Штрафы', 'Налоги', 'Кредиты']
        importance = model.feature_importances_[[0, 1, 3, 4, 5]]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importance, y=factors, palette=['#00A859'] * len(factors))
        ax.set_xlabel("Важность фактора", fontsize=12)
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)

# Вторая страница: Залоговый кредит
elif page == "Залоговый кредит":
    st.markdown("<h1 style='font-weight: 700;'>Credit Coach BCC</h1>", unsafe_allow_html=True)
    st.markdown("""
    <h3 class='center-text'>
    Рассчитайте условия залогового кредита
    </h3>
    """, unsafe_allow_html=True)

    with st.form("collateral_form"):
        st.markdown("**Данные о залоге**")
        collateral_value = st.number_input("Оценочная стоимость залога (₸)", min_value=1000000, value=5000000, step=100000)
        loan_amount = st.number_input("Сумма кредита (₸)", min_value=100000, value=2500000, step=10000)
        loan_term_months = st.slider("Срок кредита (месяцы)", 3, 120, 36)

        st.markdown("**Ваши данные**")
        iin = st.text_input("ИИН (12 цифр, обязательно)", max_chars=12)
        if not iin or len(iin) != 12 or not iin.isdigit():
            st.error("Пожалуйста, введите корректный ИИН (12 цифр).")

        cols = st.columns(2)
        with cols[0]:
            income = st.number_input("Официальная заработная плата (₸)", min_value=0, value=st.session_state.client_data["income"], step=1000)
        with cols[1]:
            loan_payments = st.number_input("Платежи по текущим кредитам (₸/месяц)", min_value=0, value=st.session_state.client_data["loan_payments"], step=1000)

        age = st.number_input("Ваш возраст", min_value=18, value=30, step=1)

        submitted = st.form_submit_button("Рассчитать")
        if submitted and (not iin or len(iin) != 12 or not iin.isdigit()):
            st.error("Пожалуйста, заполните поле ИИН корректно.")
            submitted = False

    if submitted:
        max_loan_amount, monthly_payment, approved, probability, error = collateral_loan_calculation(
            collateral_value, loan_amount, loan_term_months, income, loan_payments, age, iin
        )

        st.header("Результаты расчёта")
        if error:
            st.error(error)
        else:
            if approved:
                st.markdown("**Кредит может быть одобрен ✅**", unsafe_allow_html=True)
            else:
                st.markdown("**Кредит не одобрен ❌**", unsafe_allow_html=True)

            color_class = "red" if probability < 60 else ""
            st.markdown(f"""
                <div class="approval-circle {color_class}">{probability:.0f}%</div>
                <p class="approval-text {color_class}">Вероятность одобрения</p>
            """, unsafe_allow_html=True)

            st.metric("Максимальная сумма кредита (₸)", f"{max_loan_amount:.2f}")
            st.metric("Ежемесячный платёж (₸)", f"{monthly_payment:.2f}")
            st.markdown(f"**Процентная ставка**: 22,5% годовых")

            button_color = "#FF0000" if probability < 60 else "#00A859"
            st.markdown(f"""
                <style>
                .stButton>button {{
                    background-color: {button_color} !important;
                    transition: background-color 0.3s;
                }}
                </style>
            """, unsafe_allow_html=True)
            if approved:
                st.link_button("Оформить кредит онлайн", "https://cabinet.bcc.kz/loans/apply")
            else:
                st.subheader("Рекомендации")
                free_balance = income - loan_payments
                max_payment = free_balance * 0.4
                new_term = int(loan_amount / max_payment) + 1 if max_payment > 0 else loan_term_months
                new_amount = max_payment * loan_term_months
                st.write(f"- Попробуйте увеличить срок кредита до {new_term} месяцев, чтобы снизить ежемесячный платёж.")
                st.write(f"- Или уменьшите сумму кредита до {new_amount:.2f} ₸, чтобы уложиться в ваш бюджет.")
                st.link_button("Подобрать другой кредит", "https://bcc.kz/loans")

# Третья страница: Кредиты для бизнеса
elif page == "Кредиты для бизнеса":
    st.markdown("<h1 style='font-weight: 700;'>Credit Coach BCC</h1>", unsafe_allow_html=True)
    st.markdown("""
    <h3 class='center-text'>
    Рассчитайте условия кредита для бизнеса
    </h3>
    """, unsafe_allow_html=True)

    with st.form("business_form"):
        st.markdown("**Данные о бизнесе**")
        business_type = st.selectbox("Тип бизнеса", ["ИП", "ТОО"])
        purpose = st.selectbox("Цель кредита", ["Пополнение оборотных средств", "Инвестиции"])
        loan_amount = st.number_input("Сумма кредита (₸)", min_value=150000, value=5000000, step=10000)
        loan_term_months = st.slider("Срок кредита (месяцы)", 6, 120, 36)

        st.markdown("**Финансовые данные**")
        iin = st.text_input("ИИН/БИН (12 цифр, обязательно)", max_chars=12)
        if not iin or len(iin) != 12 or not iin.isdigit():
            st.error("Пожалуйста, введите корректный ИИН/БИН (12 цифр).")

        cols = st.columns(2)
        with cols[0]:
            income = st.number_input("Доходы бизнеса (₸/месяц)", min_value=0, value=1000000, step=1000)
        with cols[1]:
            loan_payments = st.number_input("Платежи по текущим кредитам (₸/месяц)", min_value=0, value=20000, step=1000)

        submitted = st.form_submit_button("Рассчитать")
        if submitted and (not iin or len(iin) != 12 or not iin.isdigit()):
            st.error("Пожалуйста, заполните поле ИИН/БИН корректно.")
            submitted = False

    if submitted:
        monthly_payment, approved, probability, error = business_loan_calculation(
            loan_amount, loan_term_months, income, loan_payments, business_type, purpose, iin
        )

        st.header("Результаты расчёта")
        if error:
            st.error(error)
        else:
            if approved:
                st.markdown("**Кредит может быть одобрен ✅**", unsafe_allow_html=True)
            else:
                st.markdown("**Кредит не одобрен ❌**", unsafe_allow_html=True)

            color_class = "red" if probability < 60 else ""
            st.markdown(f"""
                <div class="approval-circle {color_class}">{probability:.0f}%</div>
                <p class="approval-text {color_class}">Вероятность одобрения</p>
            """, unsafe_allow_html=True)

            st.metric("Ежемесячный платёж (₸)", f"{monthly_payment:.2f}")
            st.markdown(f"**Процентная ставка**: 24,45% годовых")

            button_color = "#FF0000" if probability < 60 else "#00A859"
            st.markdown(f"""
                <style>
                .stButton>button {{
                    background-color: {button_color} !important;
                    transition: background-color 0.3s;
                }}
                </style>
            """, unsafe_allow_html=True)
            if approved:
                st.link_button("Оформить кредит онлайн", "https://cabinet.bcc.kz/loans/apply")
            else:
                st.subheader("Рекомендации")
                free_balance = income - loan_payments
                max_payment = free_balance * 0.4
                new_term = int(loan_amount / max_payment) + 1 if max_payment > 0 else loan_term_months
                new_amount = max_payment * loan_term_months
                st.write(f"- Попробуйте увеличить срок кредита до {new_term} месяцев, чтобы снизить ежемесячный платёж.")
                st.write(f"- Или уменьшите сумму кредита до {new_amount:.2f} ₸, чтобы уложиться в ваш бюджет.")
                st.link_button("Подобрать другой кредит", "https://bcc.kz/loans")

# Карта отделений и контакты
st.markdown("---")
st.subheader("Найдите ближайшее отделение БЦК")
st.markdown("""
<iframe src="https://www.google.com/maps/embed?pb=!1m16!1m12!1m3!1d159220.4096707107!2d76.81924257070312!3d43.2389498!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!2m1!1s%D0%91%D0%B0%D0%BD%D0%BA%20%D0%A6%D0%B5%D0%BD%D1%82%D1%80%D0%9A%D1%80%D0%B5%D0%B4%D0%B8%D1%82%20%D0%90%D0%BB%D0%BC%D0%B0%D1%82%D1%8B!5e0!3m2!1sru!2skz!4v1732071234567!5m2!1sru!2skz" 
width="100%" height="450" style="border:0;" allowfullscreen="" loading="lazy"></iframe>
""", unsafe_allow_html=True)

st.subheader("Контакты БЦК")
st.markdown("""
- **Колл-центр**: 505 (с мобильного бесплатно)  
- **Телефон для звонков**: +7 (727) 244-35-55  
- **Email**: info@bcc.kz  
- **Официальный сайт**: [www.bcc.kz](https://www.bcc.kz)
""")

# Чат-бот
st.markdown("""
<div class="chatbot-button" id="chatbot-button" onclick="toggleChatbot()">
    Чат
</div>

<div class="chatbot" id="chatbot-window" style="display: none;">
    <div class="chatbot-header">
        ИИ-ассистент БЦК
        <button onclick="toggleChatbot()" style="float: right; background: none; border: none; color: white; cursor: pointer;">✖</button>
    </div>
    <div class="chatbot-body" id="chatbot-body">
        <div class="chatbot-message bot">Здравствуйте! Чем могу помочь?</div>
    </div>
    <div class="chatbot-input">
        <input type="text" id="chatbot-input" placeholder="Введите сообщение...">
        <button onclick="sendMessage()">Отправить</button>
    </div>
</div>

<script>
function toggleChatbot() {
    var chatbotWindow = document.getElementById("chatbot-window");
    var chatbotButton = document.getElementById("chatbot-button");
    if (chatbotWindow.style.display === "none") {
        chatbotWindow.style.display = "block";
        chatbotButton.style.display = "none";
    } else {
        chatbotWindow.style.display = "none";
        chatbotButton.style.display = "flex";
    }
}

function sendMessage() {
    var input = document.getElementById("chatbot-input");
    var message = input.value.trim();
    if (message === "") return;

    var chatBody = document.getElementById("chatbot-body");
    chatBody.innerHTML += '<div class="chatbot-message user">' + message + '</div>';

    var response = "Спасибо за ваш вопрос! Пока я могу только направить вас в колл-центр: 505.";
    chatBody.innerHTML += '<div class="chatbot-message bot">' + response + '</div>';

    input.value = "";
    chatBody.scrollTop = chatBody.scrollHeight;
}

document.getElementById("chatbot-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});
</script>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("© 2025 Банк ЦентрКредит. Все права защищены.")