"""Simple Persian RTL demo UI: train once, then test personalization for one user."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import streamlit as st

from data.dataset import load_ratings
from recommender.model import MovieRecommender, split_ratings
from recommender.recommend import recommend_movies

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "movie_recommender.joblib"
TEST_PATH = ARTIFACTS / "test_ratings.csv"
SUMMARY_PATH = ARTIFACTS / "training_summary.json"
N_COMPONENTS = 20

st.set_page_config(
    page_title="MovieMind AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Vazirmatn', sans-serif;
      }
      .stApp, [data-testid="stMain"] { direction: rtl; }
      [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { text-align: right; }
      [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,.16);
        border-radius: 14px;
        padding: .8rem 1rem;
      }
      .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 20px;
        background: linear-gradient(120deg, #111827 0%, #1e3a5f 65%, #164e63 100%);
        color: #f8fafc;
        margin-bottom: 1rem;
      }
      .hero h1 { color: white; margin: 0 0 .35rem 0; font-size: 1.9rem; }
      .hero p { color: #cbd5e1; margin: 0; }
      .stButton button, .stDownloadButton button {
        border-radius: 10px;
        min-height: 2.7rem;
      }
      div[data-testid="stProgress"] { direction: ltr; }
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


def predict_rating(model, user_id, movie_title):
    if (
        user_id in model.predicted_ratings.index
        and movie_title in model.predicted_ratings.columns
    ):
        return float(model.predicted_ratings.loc[user_id, movie_title]), "پیش‌بینی SVD"
    return float(model.global_mean), "میانگین آموزش (جایگزین)"


if "model" not in st.session_state:
    st.session_state.model = load_model()
if "test_ratings" not in st.session_state:
    st.session_state.test_ratings = load_test_ratings()
if "training_summary" not in st.session_state:
    try:
        st.session_state.training_summary = (
            json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            if SUMMARY_PATH.exists() else None
        )
    except (OSError, json.JSONDecodeError):
        st.session_state.training_summary = None

model = st.session_state.model
summary = st.session_state.training_summary

st.markdown(
    """
    <div class="hero">
      <h1>🎬 MovieMind AI</h1>
      <p>آموزش مدل پیشنهاددهنده و نمایش عملی شخصی‌سازی برای یک کاربر</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is not None:
    st.success("مدل آموزش‌دیده آماده استفاده است.")
else:
    st.warning("برای شروع، ابتدا مدل را در بخش «آموزش مدل» آموزش دهید.")

train_tab, demo_tab = st.tabs(["۱. آموزش مدل", "۲. آزمایش شخصی‌سازی برای یک کاربر"])

with train_tab:
    st.subheader("آموزش مدل پیشنهاد فیلم")
    st.write(
        "برای آماده‌سازی سیستم فقط یک کار لازم است: دکمه زیر را بزنید. "
        "برنامه داده‌ها را دریافت می‌کند، آن‌ها را نصف می‌کند، مدل SVD را آموزش می‌دهد "
        "و نتیجه را ذخیره می‌کند."
    )

    st.markdown("**تنظیمات ثابت پروژه**")
    p1, p2, p3 = st.columns(3)
    p1.metric("دیتاست", "MovieLens 100K")
    p2.metric("تقسیم داده", "۵۰٪ آموزش / ۵۰٪ آزمون")
    p3.metric("الگوریتم", "SVD با ۲۰ مؤلفه")

    if st.button("شروع آموزش مدل", type="primary", use_container_width=True):
        start = perf_counter()
        progress = st.progress(0, text="مرحله ۱ از ۴ — دریافت داده‌ها")
        status = st.empty()
        try:
            status.info("در حال دریافت MovieLens 100K؛ در اجرای اول ممکن است دانلود انجام شود.")
            ratings = get_ratings()
            progress.progress(20, text="مرحله ۱ از ۴ — داده‌ها آماده شد")

            train_data, test_data = split_ratings(ratings, test_size=0.5)
            status.info(
                f"مرحله ۲ از ۴ — تقسیم داده انجام شد: "
                f"{len(train_data):,} رکورد آموزش و {len(test_data):,} رکورد آزمون."
            )
            progress.progress(40, text="مرحله ۲ از ۴ — تقسیم داده‌ها")

            status.info("مرحله ۳ از ۴ — مدل SVD در حال یادگیری الگوهای کاربران و فیلم‌هاست.")
            trained_model = MovieRecommender(n_components=N_COMPONENTS).fit(train_data)
            progress.progress(75, text="مرحله ۳ از ۴ — آموزش مدل تمام شد")

            status.info("مرحله ۴ از ۴ — محاسبه دقت روی داده‌هایی که مدل ندیده است.")
            metrics = trained_model.evaluate(test_data)
            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            trained_model.save(MODEL_PATH)
            test_data.to_csv(TEST_PATH, index=False)

            elapsed = perf_counter() - start
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
            progress.progress(100, text="آموزش کامل شد")
            status.success("مدل آموزش داده شد و فایل‌های لازم برای آزمایش کاربر ذخیره شدند.")
            st.rerun()
        except Exception as exc:
            status.error(f"آموزش انجام نشد: {exc}")
            st.exception(exc)

    if summary:
        st.divider()
        st.subheader("نتیجه آخرین آموزش")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("RMSE", f"{summary['rmse']:.4f}")
        r2.metric("MAE", f"{summary['mae']:.4f}")
        r3.metric("زمان آموزش", f"{summary['elapsed_seconds']:.1f} ثانیه")
        r4.metric("رکوردهای آزمون", f"{summary['test_count']:,}")
        st.caption(
            f"از {summary['ratings']:,} امتیاز کل، {summary['train_count']:,} امتیاز برای آموزش "
            f"و {summary['test_count']:,} امتیاز برای آزمون کنار گذاشته شد."
        )
        with st.expander("توضیح کوتاه معیارها"):
            st.write(
                "**RMSE:** خطای پیش‌بینی را با جریمه بیشتر برای خطاهای بزرگ نشان می‌دهد. "
                "**MAE:** میانگین فاصله امتیاز پیش‌بینی‌شده تا امتیاز واقعی است. "
                "در هر دو معیار مقدار کمتر بهتر است."
            )

with demo_tab:
    st.subheader("آزمایش شخصی‌سازی برای یک کاربر")
    st.write(
        "یک کاربر انتخاب کنید. سیستم ابتدا سلیقه او را از امتیازهای بخش آموزش می‌خواند؛ "
        "سپس امتیازهای کنارگذاشته‌شده همان کاربر را پیش‌بینی می‌کند و فیلم‌های جدید پیشنهاد می‌دهد."
    )

    model = st.session_state.model
    test_data = st.session_state.test_ratings

    if model is None or model.user_item is None:
        st.info("ابتدا از بخش «آموزش مدل» مدل را آموزش دهید.")
    else:
        available_users = set(model.user_item.index.astype(str))
        if test_data is not None:
            test_users = set(test_data["user_id"].astype(str))
            demo_users = sorted(available_users.intersection(test_users))
        else:
            demo_users = sorted(available_users)

        if not demo_users:
            st.warning("کاربری برای آزمایش پیدا نشد. آموزش مدل را دوباره اجرا کنید.")
        else:
            if test_data is not None:
                test_counts = test_data[test_data["user_id"].isin(demo_users)].groupby("user_id").size()
                default_user = str(test_counts.idxmax()) if not test_counts.empty else demo_users[0]
            else:
                train_counts = model.user_item.notna().sum(axis=1)
                default_user = str(train_counts.idxmax())

            selected_user = st.selectbox(
                "شناسه کاربر برای نمایش",
                demo_users,
                index=demo_users.index(default_user),
            )

            history = model.user_item.loc[selected_user].dropna().sort_values(ascending=False)
            rated_titles = set(history.index.astype(str))
            h1, h2, h3 = st.columns(3)
            h1.metric("تعداد فیلم‌های شناخته‌شده", f"{len(history):,}")
            h2.metric("میانگین امتیازهای کاربر", f"{history.mean():.2f} از ۵")
            h3.metric("فیلم‌های قابل پیشنهاد", f"{len(model.predicted_ratings.columns) - len(rated_titles):,}")

            st.markdown("#### سلیقه‌ای که مدل از کاربر یاد گرفته است")
            liked = history[history >= 4.0].head(8)
            disliked = history[history <= 2.0].head(5)
            left, right = st.columns(2)
            with left:
                st.markdown("**فیلم‌هایی که امتیاز بالا گرفته‌اند**")
                if liked.empty:
                    st.caption("در داده‌های آموزش این کاربر امتیاز ۴ یا بالاتر ثبت نشده است.")
                else:
                    liked_view = liked.rename("امتیاز کاربر").to_frame()
                    st.dataframe(liked_view, use_container_width=True)
            with right:
                st.markdown("**فیلم‌هایی که امتیاز پایین گرفته‌اند**")
                if disliked.empty:
                    st.caption("در داده‌های آموزش این کاربر امتیاز ۲ یا پایین‌تر ثبت نشده است.")
                else:
                    disliked_view = disliked.rename("امتیاز کاربر").to_frame()
                    st.dataframe(disliked_view, use_container_width=True)

            st.markdown("#### آزمون روی فیلم‌هایی که از آموزش کنار گذاشته شده‌اند")
            if test_data is None:
                st.warning("فایل داده آزمون ذخیره نشده است. یک بار دیگر آموزش را اجرا کنید تا آزمون شخصی‌سازی فعال شود.")
            else:
                user_test = test_data[test_data["user_id"].astype(str) == selected_user].copy()
                if user_test.empty:
                    st.info("برای این کاربر رکورد آزمون موجود نیست.")
                else:
                    actual_values = []
                    predicted_values = []
                    sources = []
                    for row in user_test.itertuples(index=False):
                        prediction, source = predict_rating(model, selected_user, row.movie_title)
                        actual_values.append(float(row.user_rating))
                        predicted_values.append(prediction)
                        sources.append(source)

                    user_test["امتیاز واقعی"] = actual_values
                    user_test["امتیاز پیش‌بینی‌شده"] = predicted_values
                    user_test["خطای مطلق"] = np.abs(
                        user_test["امتیاز واقعی"] - user_test["امتیاز پیش‌بینی‌شده"]
                    )
                    user_test["روش پیش‌بینی"] = sources
                    user_rmse = float(np.sqrt(np.mean(
                        (user_test["امتیاز واقعی"] - user_test["امتیاز پیش‌بینی‌شده"]) ** 2
                    )))
                    user_mae = float(user_test["خطای مطلق"].mean())

                    e1, e2, e3 = st.columns(3)
                    e1.metric("تعداد امتیازهای آزمون کاربر", f"{len(user_test):,}")
                    e2.metric("RMSE این کاربر", f"{user_rmse:.3f}")
                    e3.metric("MAE این کاربر", f"{user_mae:.3f}")
                    st.dataframe(
                        user_test[[
                            "movie_title", "امتیاز واقعی", "امتیاز پیش‌بینی‌شده",
                            "خطای مطلق", "روش پیش‌بینی"
                        ]].rename(columns={"movie_title": "فیلم"}),
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.caption(
                        "امتیاز واقعی این جدول هنگام آموزش در اختیار مدل نبوده است. "
                        "پس از آموزش، پیش‌بینی مدل با امتیاز واقعی مقایسه می‌شود."
                    )
                    st.download_button(
                        "دریافت نتیجه آزمون این کاربر (CSV)",
                        user_test.to_csv(index=False).encode("utf-8-sig"),
                        f"user_{selected_user}_heldout_test.csv",
                        "text/csv",
                    )

            st.markdown("#### فیلم‌هایی که سیستم به این کاربر پیشنهاد می‌دهد")
            if st.button("تولید پیشنهادهای شخصی این کاربر", type="primary", use_container_width=True):
                try:
                    st.session_state.demo_recommendations = recommend_movies(
                        model, selected_user, top_n=10
                    )
                    st.session_state.demo_recommendation_user = selected_user
                except Exception as exc:
                    st.error(f"تولید پیشنهاد ناموفق بود: {exc}")

            recommendations = st.session_state.get("demo_recommendations")
            if (
                recommendations is not None
                and st.session_state.get("demo_recommendation_user") == selected_user
            ):
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
                st.bar_chart(
                    recommendations.set_index("movie_title")["predicted_rating"]
                )
                st.download_button(
                    "دریافت فهرست پیشنهادها (CSV)",
                    recommendations.to_csv(index=False).encode("utf-8-sig"),
                    f"user_{selected_user}_recommendations.csv",
                    "text/csv",
                )
                st.caption(
                    "فیلم‌های امتیازداده‌شده در بخش آموزش از پیشنهادها حذف شده‌اند. "
                    "امتیاز تخمینی، خروجی مدل است و تضمین علاقه‌مندی واقعی کاربر نیست."
                )

st.divider()
st.caption("MovieMind AI | MovieLens 100K | TruncatedSVD")
