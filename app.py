"""Streamlit dashboard for the MovieRecommendation project."""

from pathlib import Path

import pandas as pd
import streamlit as st

from data.dataset import load_ratings
from recommender.model import MovieRecommender, split_ratings
from recommender.recommend import recommend_movies

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "movie_recommender.joblib"

st.set_page_config(
    page_title="MovieMind AI | Movie Recommender",
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
      .stApp { direction: rtl; }
      [data-testid="stSidebar"] { direction: rtl; }
      [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        text-align: right;
      }
      .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #111827 0%, #1e3a5f 65%, #164e63 100%);
        color: #f8fafc;
        margin-bottom: 1.2rem;
      }
      .hero h1 { color: #fff; font-size: 2rem; margin: 0 0 .45rem 0; }
      .hero p { color: #cbd5e1; margin: 0; }
      .section-note { color: #64748b; font-size: .9rem; }
      div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        padding: 1rem 1.1rem;
        border: 1px solid rgba(128,128,128,.16);
        border-radius: 15px;
      }
      .stButton button { border-radius: 11px; min-height: 2.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_ratings():
    return load_ratings()


def load_saved_model():
    if MODEL_PATH.exists():
        try:
            return MovieRecommender.load(MODEL_PATH)
        except Exception as exc:
            st.sidebar.warning(f"مدل ذخیره‌شده قابل بارگذاری نیست: {exc}")
    return None


if "model" not in st.session_state:
    st.session_state.model = load_saved_model()
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "dataset_stats" not in st.session_state:
    st.session_state.dataset_stats = None

model = st.session_state.model
st.markdown(
    """
    <div class="hero">
      <h1>🎬 MovieMind AI</h1>
      <p>سامانه هوشمند پیشنهاد فیلم با یادگیری ماشین و فیلترسازی مشارکتی SVD</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("MovieMind AI")
    st.caption("پنل کنترل سیستم پیشنهاددهنده")
    st.divider()
    st.markdown("**وضعیت مدل**")
    if model is not None:
        st.success("مدل آموزش‌دیده بارگذاری شده است")
    else:
        st.warning("هنوز مدلی بارگذاری نشده است")
    st.caption("دیتاست: MovieLens 100K")
    st.caption("روش ارزیابی: تقسیم تصادفی ۵۰/۵۰")
    st.divider()
    st.caption("نسخه نمایشی دانشگاهی")

tab_dashboard, tab_training, tab_recommend = st.tabs(
    ["داشبورد", "آموزش و ارزیابی", "پیشنهاد فیلم"]
)

with tab_dashboard:
    st.subheader("نمای کلی پروژه")
    st.markdown(
        '<p class="section-note">آمار و معیارها پس از بارگذاری داده و اجرای آموزش نمایش داده می‌شوند.</p>',
        unsafe_allow_html=True,
    )
    if st.session_state.dataset_stats:
        stats = st.session_state.dataset_stats
        c1, c2, c3 = st.columns(3)
        c1.metric("تعداد امتیازها", f"{stats['ratings']:,}")
        c2.metric("تعداد کاربران", f"{stats['users']:,}")
        c3.metric("تعداد فیلم‌ها", f"{stats['movies']:,}")
    else:
        st.info("برای دریافت آمار واقعی دیتاست، به بخش «آموزش و ارزیابی» بروید و آموزش را اجرا کنید.")

    if st.session_state.metrics:
        metrics = st.session_state.metrics
        st.subheader("نتایج آخرین ارزیابی")
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", f"{metrics['rmse']:.4f}")
        m2.metric("MAE", f"{metrics['mae']:.4f}")
        m3.metric("تعداد نمونه‌های آزمون", f"{metrics['test_count']:,}")
        st.caption("این معیارها از داده‌های آزمونی محاسبه شده‌اند که در آموزش مدل استفاده نشده‌اند.")
    elif model is not None:
        st.info("مدل ذخیره‌شده بارگذاری شد؛ برای مشاهده معیارهای ارزیابی، آموزش را از همین پنل اجرا کنید.")

    st.subheader("مراحل کار سیستم")
    steps = pd.DataFrame(
        {
            "مرحله": ["بارگذاری داده", "تقسیم داده", "آموزش SVD", "ارزیابی", "پیشنهاد فیلم"],
            "شرح": [
                "دریافت MovieLens 100K",
                "۵۰٪ آموزش و ۵۰٪ آزمون",
                "یادگیری الگوهای کاربران و فیلم‌ها",
                "محاسبه RMSE و MAE روی داده آزمون",
                "رتبه‌بندی فیلم‌های دیده‌نشده",
            ],
        }
    )
    st.dataframe(steps, hide_index=True, use_container_width=True)

with tab_training:
    st.subheader("آموزش و ارزیابی مدل")
    st.write("داده‌ها با بذر ثابت ۴۲ به دو بخش مستقل تقسیم می‌شوند. مدل فقط با بخش آموزش برازش می‌شود.")
    left, right = st.columns([2, 1])
    with left:
        st.markdown("**تقسیم داده‌ها**")
        st.progress(0.5, text="۵۰٪ داده آموزش")
        st.progress(0.5, text="۵۰٪ داده آزمون")
    with right:
        st.metric("تعداد مؤلفه‌های SVD", "20")
        st.metric("نسبت آموزش / آزمون", "50 / 50")

    if st.button("شروع آموزش و ارزیابی", type="primary", use_container_width=True):
        try:
            with st.spinner("در حال بارگذاری MovieLens 100K..."):
                ratings = get_ratings()
            st.session_state.dataset_stats = {
                "ratings": len(ratings),
                "users": ratings["user_id"].nunique(),
                "movies": ratings["movie_title"].nunique(),
            }
            train_data, test_data = split_ratings(ratings, test_size=0.5)
            progress = st.progress(0, text="آماده‌سازی داده‌ها")
            with st.spinner("در حال آموزش مدل SVD..."):
                trained_model = MovieRecommender(n_components=20).fit(train_data)
                progress.progress(75, text="ارزیابی روی داده‌های آزمون")
                metrics = trained_model.evaluate(test_data)
                trained_model.save(MODEL_PATH)
            progress.progress(100, text="آموزش و ارزیابی کامل شد")
            st.session_state.model = trained_model
            st.session_state.metrics = metrics
            st.success("مدل با موفقیت آموزش داده و ذخیره شد.")
            a, b, c, d = st.columns(4)
            a.metric("داده آموزش", f"{len(train_data):,}")
            b.metric("داده آزمون", f"{len(test_data):,}")
            c.metric("RMSE", f"{metrics['rmse']:.4f}")
            d.metric("MAE", f"{metrics['mae']:.4f}")
            st.caption(f"مسیر ذخیره مدل: {MODEL_PATH}")
        except Exception as exc:
            st.error(f"اجرای آموزش ناموفق بود: {exc}")

with tab_recommend:
    st.subheader("پیشنهاد فیلم برای کاربر")
    model = st.session_state.model
    if model is None or model.user_item is None:
        st.warning("ابتدا مدل را در بخش «آموزش و ارزیابی» آموزش دهید یا یک مدل ذخیره‌شده معتبر داشته باشید.")
    else:
        users = sorted(model.user_item.index.astype(str).tolist())
        col_user, col_count = st.columns([2, 1])
        with col_user:
            selected_user = st.selectbox("شناسه کاربر", users)
        with col_count:
            top_n = st.selectbox("تعداد پیشنهادها", [5, 10, 15, 20], index=1)
        if st.button("دریافت پیشنهادها", type="primary", use_container_width=True):
            try:
                result = recommend_movies(model, selected_user, top_n)
                if result.empty:
                    st.info("برای این کاربر فیلم دیده‌نشده‌ای در مجموعه آموزش پیدا نشد.")
                else:
                    st.success(f"{len(result)} فیلم پیشنهادی برای کاربر {selected_user}")
                    display = result.copy()
                    display["predicted_rating"] = display["predicted_rating"].round(2)
                    display.index = range(1, len(display) + 1)
                    st.dataframe(
                        display.rename(
                            columns={
                                "user_id": "شناسه کاربر",
                                "movie_title": "نام فیلم",
                                "predicted_rating": "امتیاز پیش‌بینی‌شده",
                            }
                        ),
                        use_container_width=True,
                    )
                    st.download_button(
                        "دانلود پیشنهادها به صورت CSV",
                        data=result.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"recommendations_user_{selected_user}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            except Exception as exc:
                st.error(f"تولید پیشنهاد ناموفق بود: {exc}")

st.divider()
st.caption("MovieMind AI • SVD Collaborative Filtering • MovieLens 100K")
