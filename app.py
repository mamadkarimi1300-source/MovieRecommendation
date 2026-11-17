"""MovieMind AI — Persian RTL dashboard with Recommender, Simulation and Laboratory areas."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import streamlit as st

from data.dataset import load_ratings
from recommender.model import MovieRecommender, split_ratings
from recommender.recommend import recommend_movies
from simulation.config import SimulationConfig
from simulation.engine import SimulationEngine
from laboratory.engine import LaboratoryEngine
from laboratory.experiments.store import ExperimentStore

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "movie_recommender.joblib"
TEST_PATH = ARTIFACTS / "test_ratings.csv"
SUMMARY_PATH = ARTIFACTS / "training_summary.json"
N_COMPONENTS = 20

st.set_page_config(
    page_title="MovieMind AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Vazirmatn', sans-serif;
      }
      .stApp, [data-testid="stMain"], [data-testid="stSidebar"] { direction: rtl; }
      [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { text-align: right; }
      [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,.16);
        border-radius: 15px;
        padding: .85rem 1rem;
      }
      .brand {
        padding: 1.15rem 1.25rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #111827 0%, #1e3a5f 65%, #164e63 100%);
        color: #f8fafc;
        margin: .25rem 0 1.1rem 0;
      }
      .brand h1 { color: white; margin: 0 0 .3rem 0; font-size: 1.8rem; }
      .brand p { color: #cbd5e1; margin: 0; font-size: .95rem; }
      .section-kicker {
        color: var(--text-color); opacity: .65; font-size: .82rem;
        font-weight: 700; letter-spacing: .02em; margin-bottom: .2rem;
      }
      .panel {
        border: 1px solid rgba(128,128,128,.2);
        border-radius: 16px; padding: 1rem 1.1rem;
        background: var(--secondary-background-color);
        min-height: 112px;
      }
      .panel-title { font-weight: 700; margin-bottom: .35rem; }
      .panel-copy { opacity: .72; font-size: .9rem; line-height: 1.8; }
      .stButton button, .stDownloadButton button {
        border-radius: 11px; min-height: 2.65rem;
      }
      div[data-testid="stProgress"] { direction: ltr; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label { padding: .25rem 0; }
      .status-line { font-size: .88rem; opacity: .75; }
      [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1500px; padding-top: 1.5rem; padding-bottom: 3rem;
      }
      [data-testid="stSidebar"] { border-left: 1px solid rgba(128,128,128,.18); }
      [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
      [data-testid="stRadio"] > div { gap: .45rem; }
      [data-testid="stRadio"] label {
        border: 1px solid rgba(128,128,128,.16); border-radius: 12px;
        padding: .55rem .75rem; background: var(--secondary-background-color);
      }
      [data-testid="stMetric"] { min-height: 112px; display: flex; flex-direction: column; justify-content: center; }
      [data-testid="stMetricValue"] { font-weight: 800; letter-spacing: -.03em; }
      [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid rgba(128,128,128,.16); border-radius: 13px; overflow: hidden;
      }
      [data-testid="stTabs"] button { font-weight: 700; }
      .page-heading { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin:.25rem 0 1rem; }
      .page-heading small { opacity:.62; font-weight:700; letter-spacing:.08em; }
      .chart-heading { font-size:1.02rem; font-weight:800; margin:.4rem 0 .2rem; }
      .muted { opacity:.68; font-size:.88rem; }
      .brand { box-shadow: 0 12px 32px rgba(15,23,42,.12); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_ratings():
    return load_ratings()


def load_model():
    if MODEL_PATH.exists():
        try:
            return MovieRecommender.load(MODEL_PATH)
        except Exception:
            return None
    return None


def load_test_ratings():
    if TEST_PATH.exists():
        try:
            frame = pd.read_csv(TEST_PATH)
            frame["user_id"] = frame["user_id"].astype(str)
            frame["movie_title"] = frame["movie_title"].astype(str)
            frame["user_rating"] = pd.to_numeric(frame["user_rating"])
            return frame
        except (OSError, ValueError, pd.errors.ParserError):
            return None
    return None


def load_summary():
    try:
        return (
            json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            if SUMMARY_PATH.exists() else None
        )
    except (OSError, json.JSONDecodeError):
        return None


def predict_rating(model, user_id, movie_title):
    if user_id in model.predicted_ratings.index and movie_title in model.predicted_ratings.columns:
        return float(model.predicted_ratings.loc[user_id, movie_title]), "پیش‌بینی SVD"
    return float(model.global_mean), "میانگین آموزش (جایگزین)"


if "model" not in st.session_state:
    st.session_state.model = load_model()
if "test_ratings" not in st.session_state:
    st.session_state.test_ratings = load_test_ratings()
if "training_summary" not in st.session_state:
    st.session_state.training_summary = load_summary()
if "demo_recommendations" not in st.session_state:
    st.session_state.demo_recommendations = None
if "demo_recommendation_user" not in st.session_state:
    st.session_state.demo_recommendation_user = None

model = st.session_state.model
summary = st.session_state.training_summary

st.markdown(
    """
    <div class="brand">
      <h1>🎬 MovieMind AI</h1>
      <p>محیط یکپارچه پیشنهاددهنده، شبیه‌سازی کاربران و آزمایشگاه یادگیری</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## MovieMind AI")
    st.caption("ناوبری محیط")
    page = st.radio(
        "بخش موردنظر",
        ["پیشنهاددهنده", "شبیه‌سازی", "آزمایشگاه"],
        horizontal=False,
        format_func=lambda x: {
            "پیشنهاددهنده": "🎬  پیشنهاددهنده",
            "شبیه‌سازی": "🧪  شبیه‌سازی کاربران",
            "آزمایشگاه": "🔬  آزمایشگاه یادگیری",
        }[x],
        label_visibility="collapsed",
    )
    st.divider()
    if model is not None:
        st.success("مدل SVD آماده استفاده است")
    else:
        st.warning("مدل SVD هنوز آموزش داده نشده")
    st.caption("MovieLens 100K · SVD")


def render_recommender():
    st.markdown('<div class="section-kicker">CORE ENGINE</div>', unsafe_allow_html=True)
    st.title("پیشنهاددهنده")
    st.caption("آموزش مدل فعلی و بررسی پیشنهادهای شخصی‌سازی‌شده؛ موتور SVD موجود بدون تغییر حفظ شده است.")

    train_tab, user_tab = st.tabs(["آموزش مدل", "آزمایش یک کاربر"])

    with train_tab:
        st.subheader("آموزش مدل پیشنهاد فیلم")
        a, b, c = st.columns(3)
        a.metric("دیتاست", "MovieLens 100K")
        b.metric("تقسیم داده", "۵۰٪ / ۵۰٪")
        c.metric("الگوریتم", "TruncatedSVD · 20")

        if st.button("شروع آموزش مدل", type="primary", use_container_width=True):
            started = perf_counter()
            progress = st.progress(0, text="۱ از ۴ — دریافت داده‌ها")
            status = st.empty()
            try:
                status.info("در حال دریافت MovieLens 100K؛ اجرای اول ممکن است شامل دانلود باشد.")
                ratings = get_ratings()
                progress.progress(20, text="۱ از ۴ — داده‌ها آماده شد")

                train_data, test_data = split_ratings(ratings, test_size=0.5)
                status.info(
                    f"۲ از ۴ — تقسیم داده: {len(train_data):,} آموزش و {len(test_data):,} آزمون"
                )
                progress.progress(40, text="۲ از ۴ — تقسیم داده‌ها")

                status.info("۳ از ۴ — آموزش مدل SVD")
                trained_model = MovieRecommender(n_components=N_COMPONENTS).fit(train_data)
                progress.progress(75, text="۳ از ۴ — آموزش تمام شد")

                status.info("۴ از ۴ — ارزیابی روی داده‌های کنارگذاشته‌شده")
                metrics = trained_model.evaluate(test_data)
                ARTIFACTS.mkdir(parents=True, exist_ok=True)
                trained_model.save(MODEL_PATH)
                test_data.to_csv(TEST_PATH, index=False)

                elapsed = perf_counter() - started
                new_summary = {
                    "ratings": int(len(ratings)),
                    "users": int(ratings["user_id"].nunique()),
                    "movies": int(ratings["movie_title"].nunique()),
                    "train_count": int(len(train_data)),
                    "test_count": int(len(test_data)),
                    "rmse": float(metrics["rmse"]),
                    "mae": float(metrics["mae"]),
                    "components": int(trained_model.model.n_components),
                    "elapsed_seconds": round(elapsed, 2),
                }
                SUMMARY_PATH.write_text(
                    json.dumps(new_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                st.session_state.model = trained_model
                st.session_state.test_ratings = test_data.reset_index(drop=True)
                st.session_state.training_summary = new_summary
                st.session_state.demo_recommendations = None
                st.session_state.demo_recommendation_user = None
                progress.progress(100, text="آموزش کامل شد")
                status.success("مدل و خروجی‌های ارزیابی ذخیره شدند.")
                st.rerun()
            except Exception as exc:
                status.error(f"آموزش انجام نشد: {exc}")
                st.exception(exc)

        summary = st.session_state.training_summary
        if summary:
            st.divider()
            st.subheader("آخرین نتیجه آموزش")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("RMSE", f"{summary['rmse']:.4f}")
            m2.metric("MAE", f"{summary['mae']:.4f}")
            m3.metric("زمان آموزش", f"{summary['elapsed_seconds']:.1f} ثانیه")
            m4.metric("رکورد آزمون", f"{summary['test_count']:,}")
            st.caption(
                f"{summary['ratings']:,} امتیاز کل · {summary['train_count']:,} آموزش · "
                f"{summary['test_count']:,} آزمون"
            )
            st.markdown('<div class="chart-heading">ترکیب داده‌های آموزش و آزمون</div>', unsafe_allow_html=True)
            split_frame = pd.DataFrame(
                {"تعداد رکورد": [summary["train_count"], summary["test_count"]]},
                index=["آموزش", "آزمون"],
            )
            st.bar_chart(split_frame, use_container_width=True)
            st.markdown('<div class="muted">این نمودار اندازه واقعی دو بخش داده را پس از آخرین آموزش نشان می‌دهد.</div>', unsafe_allow_html=True)

    with user_tab:
        model = st.session_state.model
        test_data = st.session_state.test_ratings
        if model is None or model.user_item is None:
            st.info("برای فعال شدن این بخش، ابتدا مدل را در زبانه «آموزش مدل» آموزش دهید.")
            return

        available_users = set(model.user_item.index.astype(str))
        if test_data is not None:
            demo_users = sorted(available_users.intersection(set(test_data["user_id"].astype(str))))
        else:
            demo_users = sorted(available_users)
        if not demo_users:
            st.warning("کاربری برای آزمایش پیدا نشد.")
            return

        if test_data is not None:
            counts = test_data[test_data["user_id"].isin(demo_users)].groupby("user_id").size()
            default_user = str(counts.idxmax()) if not counts.empty else demo_users[0]
        else:
            default_user = str(model.user_item.notna().sum(axis=1).idxmax())

        selected_user = st.selectbox(
            "انتخاب شناسه کاربر", demo_users, index=demo_users.index(default_user)
        )
        history = model.user_item.loc[selected_user].dropna().sort_values(ascending=False)
        liked = history[history >= 4.0].head(8)
        disliked = history[history <= 2.0].head(5)

        h1, h2, h3 = st.columns(3)
        h1.metric("فیلم‌های شناخته‌شده", f"{len(history):,}")
        h2.metric("میانگین امتیاز", f"{history.mean():.2f} از ۵")
        h3.metric("فیلم‌های کاندید", f"{len(model.predicted_ratings.columns) - len(history):,}")

        st.markdown('<div class="chart-heading">توزیع امتیازهای ثبت‌شده کاربر</div>', unsafe_allow_html=True)
        rating_distribution = history.value_counts().sort_index().rename_axis("امتیاز").to_frame("تعداد فیلم")
        st.bar_chart(rating_distribution, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.markdown("**امتیازهای بالا**")
            st.dataframe(liked.rename("امتیاز").to_frame(), use_container_width=True)
        with right:
            st.markdown("**امتیازهای پایین**")
            st.dataframe(disliked.rename("امتیاز").to_frame(), use_container_width=True)

        st.divider()
        st.markdown("#### ارزیابی روی امتیازهای کنارگذاشته‌شده")
        if test_data is None:
            st.warning("فایل آزمون موجود نیست؛ آموزش را دوباره اجرا کنید.")
        else:
            user_test = test_data[test_data["user_id"].astype(str) == selected_user].copy()
            if user_test.empty:
                st.info("برای این کاربر رکورد آزمون وجود ندارد.")
            else:
                predictions = [
                    predict_rating(model, selected_user, row.movie_title)
                    for row in user_test.itertuples(index=False)
                ]
                user_test["امتیاز واقعی"] = user_test["user_rating"].astype(float)
                user_test["امتیاز پیش‌بینی‌شده"] = [x[0] for x in predictions]
                user_test["روش پیش‌بینی"] = [x[1] for x in predictions]
                errors = user_test["امتیاز واقعی"] - user_test["امتیاز پیش‌بینی‌شده"]
                e1, e2, e3 = st.columns(3)
                e1.metric("تعداد آزمون", f"{len(user_test):,}")
                e2.metric("RMSE کاربر", f"{np.sqrt(np.mean(errors ** 2)):.3f}")
                e3.metric("MAE کاربر", f"{np.mean(np.abs(errors)):.3f}")
                st.dataframe(
                    user_test[["movie_title", "امتیاز واقعی", "امتیاز پیش‌بینی‌شده", "روش پیش‌بینی"]]
                    .rename(columns={"movie_title": "فیلم"}),
                    hide_index=True,
                    use_container_width=True,
                )
                st.download_button(
                    "دریافت نتایج آزمون کاربر",
                    user_test.to_csv(index=False).encode("utf-8-sig"),
                    f"user_{selected_user}_test.csv",
                    "text/csv",
                )

        st.divider()
        if st.button("تولید پیشنهادهای شخصی", type="primary", use_container_width=True):
            try:
                st.session_state.demo_recommendations = recommend_movies(model, selected_user, top_n=10)
                st.session_state.demo_recommendation_user = selected_user
            except Exception as exc:
                st.error(f"تولید پیشنهاد ناموفق بود: {exc}")

        recommendations = st.session_state.demo_recommendations
        if recommendations is not None and st.session_state.demo_recommendation_user == selected_user:
            display = recommendations.copy()
            display["predicted_rating"] = display["predicted_rating"].round(2)
            display.index = range(1, len(display) + 1)
            st.dataframe(
                display.rename(columns={
                    "movie_title": "فیلم پیشنهادی",
                    "predicted_rating": "امتیاز تخمینی از ۵",
                    "user_id": "شناسه کاربر",
                }),
                use_container_width=True,
            )
            st.bar_chart(recommendations.set_index("movie_title")["predicted_rating"])
            st.download_button(
                "دریافت فهرست پیشنهادها",
                recommendations.to_csv(index=False).encode("utf-8-sig"),
                f"user_{selected_user}_recommendations.csv",
                "text/csv",
            )


def get_laboratory_engine():
    if "laboratory_engine" not in st.session_state:
        st.session_state.laboratory_engine = LaboratoryEngine()
    return st.session_state.laboratory_engine


def render_simulation():
    st.markdown('<div class="section-kicker">USER BEHAVIOR ENVIRONMENT</div>', unsafe_allow_html=True)
    st.title("شبیه‌سازی کاربران")
    st.caption("تولید کاربران مصنوعی، ساخت تعاملات و ارسال مستقیم رویدادها به آزمایشگاه.")

    engine = get_laboratory_engine()
    metrics = engine.metrics()
    c1, c2, c3 = st.columns(3)
    c1.metric("کاربران شبیه‌سازی", "۵۰")
    c2.metric("رویدادهای پردازش‌شده", f"{metrics['processed_events']:,}")
    c3.metric("اتصال آزمایشگاه", "متصل")

    st.subheader("پیکربندی شبیه‌سازی")
    interactions_per_user = st.slider("تعداد تعامل هر کاربر", 1, 20, 5)
    st.caption("الگوی کاربران ترکیبی است: مشاهده‌گر فعال، امتیازدهنده، کاوشگر و کاربر با ترجیحات متغیر.")
    st.caption("امتیازها مصنوعی و بر اساس ترجیحات نهفته تولید می‌شوند؛ این داده‌ها امتیاز واقعی MovieLens نیستند.")

    if st.button("شروع شبیه‌سازی ۵۰ کاربر", type="primary", use_container_width=True):
        try:
            with st.spinner("در حال تولید کاربران و انتقال رویدادها به آزمایشگاه..."):
                ratings = get_ratings()
                catalog = sorted(ratings["movie_title"].dropna().astype(str).unique().tolist())
                recommendation_titles = []
                active_model = st.session_state.model
                if active_model is not None and active_model.predicted_ratings is not None:
                    recommendation_titles = (
                        active_model.predicted_ratings.mean(axis=0)
                        .nlargest(30).index.astype(str).tolist()
                    )
                config = SimulationConfig(
                    user_count=50,
                    interactions_per_user=interactions_per_user,
                    random_seed=42,
                    include_recommendation_events=True,
                )
                simulation = SimulationEngine(config)
                events = simulation.run_batch(catalog, recommendation_titles)
                for event in events:
                    engine.consume(event)
                st.session_state.simulation_users = [
                    {
                        "user_id": user.user_id,
                        "behavior_type": user.behavior_type,
                        "activity_level": user.activity_level,
                        "exploration_rate": user.exploration_rate,
                    }
                    for user in simulation.users
                ]
                st.session_state.simulation_events = [event.to_dict() for event in events]
                st.session_state.last_simulation_config = {
                    "user_count": config.user_count,
                    "interactions_per_user": config.interactions_per_user,
                    "learning_method": "incremental",
                    "recommendation_events": bool(recommendation_titles),
                }
                st.success(f"شبیه‌سازی تمام شد؛ {len(simulation.users)} کاربر و {len(events):,} رویداد تولید و پردازش شد.")
                st.rerun()
        except Exception as exc:
            st.error(f"اجرای شبیه‌سازی ناموفق بود: {exc}")

    users = st.session_state.get("simulation_users", [])
    if users:
        st.divider()
        st.subheader("نمای تحلیلی جمعیت")
        user_frame = pd.DataFrame(users)
        behavior_counts = user_frame["behavior_type"].value_counts().rename_axis("الگوی رفتاری").to_frame("تعداد کاربران")
        left_chart, right_chart = st.columns(2)
        with left_chart:
            st.markdown('<div class="chart-heading">ترکیب الگوهای رفتاری</div>', unsafe_allow_html=True)
            st.bar_chart(behavior_counts, use_container_width=True)
        with right_chart:
            st.markdown('<div class="chart-heading">سطح فعالیت کاربران</div>', unsafe_allow_html=True)
            st.bar_chart(user_frame.set_index("user_id")[["activity_level", "exploration_rate"]], use_container_width=True)
        st.subheader("مشخصات کاربران تولیدشده")
        st.dataframe(user_frame, hide_index=True, use_container_width=True)
        events = st.session_state.get("simulation_events", [])
        st.subheader("آخرین رویدادهای تولیدشده")
        if events:
            event_frame = pd.DataFrame(events)
            if "event_type" in event_frame.columns:
                event_counts = event_frame["event_type"].value_counts().rename_axis("نوع رویداد").to_frame("تعداد")
                st.markdown('<div class="chart-heading">توزیع رویدادهای تولیدشده</div>', unsafe_allow_html=True)
                st.bar_chart(event_counts, use_container_width=True)
            st.markdown('<div class="chart-heading">جریان آخرین رویدادها</div>', unsafe_allow_html=True)
            st.dataframe(event_frame.tail(100), hide_index=True, use_container_width=True)
            st.download_button(
                "دریافت رویدادهای این اجرا (JSON)",
                json.dumps(events, ensure_ascii=False, indent=2),
                "simulation_events.json",
                "application/json",
            )
    else:
        st.info("هنوز شبیه‌سازی اجرا نشده است. با شروع آزمایش، کاربران و رویدادهای واقعیِ تولیدشده در همین صفحه نمایش داده می‌شوند.")


def render_laboratory():
    st.markdown('<div class="section-kicker">MODEL LEARNING & EXPERIMENTS</div>', unsafe_allow_html=True)
    st.title("آزمایشگاه یادگیری")
    st.caption("دریافت رویدادهای Simulation، پیش‌بینی قبل از آموزش و به‌روزرسانی تدریجی مدل.")

    engine = get_laboratory_engine()
    metrics = engine.metrics()
    a, b, c = st.columns(3)
    a.metric("روش یادگیری", "Incremental")
    b.metric("رویدادهای دریافتی", f"{metrics['processed_events']:,}")
    c.metric("به‌روزرسانی مدل", f"{metrics['model_updates']:,}")

    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE پیش‌رونده", f"{metrics['rmse']:.4f}" if metrics["rmse"] is not None else "—")
    m2.metric("MAE پیش‌رونده", f"{metrics['mae']:.4f}" if metrics["mae"] is not None else "—")
    m3.metric("امتیازهای ارزیابی‌شده", f"{metrics['rating_count']:,}")

    st.caption("معیارها به روش Prequential محاسبه می‌شوند: هر امتیاز ابتدا پیش‌بینی و ارزیابی می‌شود و سپس وارد آموزش مدل می‌شود.")
    if metrics["event_counts"]:
        st.subheader("تحلیل رویدادهای دریافت‌شده")
        event_chart = pd.Series(metrics["event_counts"], name="تعداد رویداد").sort_values(ascending=False)
        st.bar_chart(event_chart, use_container_width=True)
    lab_events = st.session_state.get("simulation_events", [])
    if lab_events:
        lab_frame = pd.DataFrame(lab_events)
        if "rating" in lab_frame.columns:
            ratings_frame = pd.to_numeric(lab_frame["rating"], errors="coerce").dropna()
            if not ratings_frame.empty:
                st.markdown('<div class="chart-heading">توزیع امتیازهای مصنوعی</div>', unsafe_allow_html=True)
                st.bar_chart(ratings_frame.value_counts().sort_index().rename_axis("امتیاز").to_frame("تعداد"), use_container_width=True)

    if st.button("ثبت نتیجه آزمایش", disabled=metrics["processed_events"] == 0):
        try:
            config = st.session_state.get("last_simulation_config", {"learning_method": "incremental"})
            record = ExperimentStore().save(config, metrics)
            st.success(f"نتیجه آزمایش {record['experiment_id']} ذخیره شد.")
        except OSError as exc:
            st.error(f"ذخیره نتیجه آزمایش انجام نشد: {exc}")

    st.divider()
    st.subheader("تاریخچه آزمایش‌ها")
    history = ExperimentStore().list()
    if history:
        rows = []
        for item in history:
            result = item.get("metrics", {})
            rows.append({
                "شناسه آزمایش": item.get("experiment_id"),
                "زمان": item.get("created_at"),
                "وضعیت": item.get("status"),
                "رویدادها": result.get("processed_events", 0),
                "به‌روزرسانی": result.get("model_updates", 0),
                "RMSE": result.get("rmse"),
                "MAE": result.get("mae"),
            })
        history_frame = pd.DataFrame(rows)
        st.dataframe(history_frame, hide_index=True, use_container_width=True)
        chart_frame = history_frame.copy()
        for metric_name in ["RMSE", "MAE"]:
            chart_frame[metric_name] = pd.to_numeric(chart_frame[metric_name], errors="coerce")
        if chart_frame[["RMSE", "MAE"]].notna().any().any():
            st.markdown('<div class="chart-heading">مقایسه خطای آزمایش‌ها</div>', unsafe_allow_html=True)
            st.line_chart(chart_frame.set_index("شناسه آزمایش")[["RMSE", "MAE"]], use_container_width=True)
    else:
        st.info("هنوز نتیجه‌ای ذخیره نشده است. پس از اجرای Simulation می‌توانید نتیجه را ثبت کنید.")

    if st.button("بازنشانی وضعیت یادگیری این نشست", type="secondary"):
        st.session_state.laboratory_engine = LaboratoryEngine()
        st.session_state.simulation_users = []
        st.session_state.simulation_events = []
        st.rerun()


if page == "پیشنهاددهنده":
    render_recommender()
elif page == "شبیه‌سازی":
    render_simulation()
else:
    render_laboratory()

st.divider()
st.caption("MovieMind AI · Recommender / Simulation / Laboratory")
