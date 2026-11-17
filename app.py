"""Detailed Persian RTL Streamlit dashboard for MovieRecommendation."""

from datetime import datetime
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
REPORT_PATH = ARTIFACTS / "training_report.json"
EVALUATION_PATH = ARTIFACTS / "evaluation_predictions.csv"

st.set_page_config(
    page_title="MovieMind AI | سامانه پیشنهاد فیلم",
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
      .stApp, [data-testid="stSidebar"], [data-testid="stMain"] { direction: rtl; }
      [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { text-align: right; }
      [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        padding: 1rem 1.1rem;
        border: 1px solid rgba(128,128,128,.16);
        border-radius: 16px;
      }
      .hero {
        padding: 1.8rem 2rem; border-radius: 22px;
        background: linear-gradient(120deg, #111827 0%, #1e3a5f 65%, #164e63 100%);
        color: #f8fafc; margin-bottom: 1.2rem;
      }
      .hero h1 { color: #fff; font-size: 2rem; margin: 0 0 .5rem 0; }
      .hero p { color: #cbd5e1; margin: 0; font-size: 1rem; }
      .eyebrow { color: #67e8f9; font-size: .8rem; font-weight: 700; }
      .section-note { color: #94a3b8; font-size: .9rem; }
      .info-panel {
        border: 1px solid rgba(128,128,128,.2); border-radius: 15px;
        padding: 1rem 1.2rem; background: var(--secondary-background-color);
      }
      .stButton button, .stDownloadButton button { border-radius: 11px; min-height: 2.7rem; }
      div[data-testid="stProgress"] { direction: ltr; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_ratings():
    return load_ratings()


def make_evaluation_details(trained_model, test_data):
    """Build per-rating predictions using the same fallback as model.evaluate."""
    actual = test_data["user_rating"].astype(float).to_numpy()
    predicted = []
    sources = []
    for row in test_data.itertuples(index=False):
        user_id = str(row.user_id)
        title = row.movie_title
        if (
            user_id in trained_model.predicted_ratings.index
            and title in trained_model.predicted_ratings.columns
        ):
            predicted.append(float(trained_model.predicted_ratings.loc[user_id, title]))
            sources.append("SVD reconstruction")
        else:
            predicted.append(float(trained_model.global_mean))
            sources.append("Training global mean (fallback)")

    details = test_data.reset_index(drop=True).copy()
    details["actual_rating"] = actual
    details["predicted_rating"] = predicted
    details["absolute_error"] = np.abs(actual - np.asarray(predicted))
    details["prediction_source"] = sources
    details["error"] = actual - np.asarray(predicted)
    return details


def build_report(ratings, train_data, test_data, trained_model, metrics, elapsed):
    matrix = trained_model.user_item
    total_cells = int(matrix.shape[0] * matrix.shape[1])
    observed_cells = int(matrix.notna().sum().sum())
    explained = getattr(trained_model.model, "explained_variance_ratio_", np.array([]))
    details = make_evaluation_details(trained_model, test_data)
    cold_users = int((~test_data["user_id"].astype(str).isin(matrix.index)).sum())
    cold_movies = int((~test_data["movie_title"].isin(matrix.columns)).sum())
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": {
            "ratings": int(len(ratings)),
            "users": int(ratings["user_id"].nunique()),
            "movies": int(ratings["movie_title"].nunique()),
            "rating_min": float(ratings["user_rating"].min()),
            "rating_max": float(ratings["user_rating"].max()),
            "rating_mean": float(ratings["user_rating"].mean()),
        },
        "split": {
            "train_ratings": int(len(train_data)),
            "test_ratings": int(len(test_data)),
            "train_ratio": len(train_data) / len(ratings),
            "test_ratio": len(test_data) / len(ratings),
            "random_state": 42,
        },
        "model": {
            "algorithm": "TruncatedSVD collaborative filtering",
            "requested_components": int(trained_model.n_components),
            "used_components": int(trained_model.model.n_components),
            "matrix_users": int(matrix.shape[0]),
            "matrix_movies": int(matrix.shape[1]),
            "matrix_density": observed_cells / total_cells if total_cells else 0.0,
            "global_mean": float(trained_model.global_mean),
            "explained_variance_ratio_sum": float(explained.sum()) if len(explained) else None,
            "fill_strategy": "Per-user mean for missing user-item ratings",
        },
        "evaluation": {
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
            "test_count": int(metrics["test_count"]),
            "fallback_user_ratings": cold_users,
            "fallback_movie_ratings": cold_movies,
            "fallback_rows": int((details["prediction_source"] != "SVD reconstruction").sum()),
        },
        "runtime_seconds": round(float(elapsed), 3),
        "artifacts": [
            str(MODEL_PATH),
            str(REPORT_PATH),
            str(EVALUATION_PATH),
        ],
    }, details


def load_saved_model():
    if MODEL_PATH.exists():
        try:
            return MovieRecommender.load(MODEL_PATH)
        except Exception as exc:
            st.sidebar.warning(f"مدل ذخیره‌شده بارگذاری نشد: {exc}")
    return None


if "model" not in st.session_state:
    st.session_state.model = load_saved_model()
if "report" not in st.session_state:
    try:
        st.session_state.report = (
            json.loads(REPORT_PATH.read_text(encoding="utf-8"))
            if REPORT_PATH.exists() else None
        )
    except (OSError, json.JSONDecodeError):
        st.session_state.report = None
if "evaluation_details" not in st.session_state:
    try:
        st.session_state.evaluation_details = (
            pd.read_csv(EVALUATION_PATH)
            if EVALUATION_PATH.exists() else None
        )
    except (OSError, pd.errors.ParserError):
        st.session_state.evaluation_details = None
if "ratings" not in st.session_state:
    st.session_state.ratings = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

model = st.session_state.model
report = st.session_state.report

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">MACHINE LEARNING • COLLABORATIVE FILTERING</div>
      <h1>🎬 MovieMind AI</h1>
      <p>آزمایشگاه تحلیل، آموزش و ارزیابی سیستم پیشنهاددهنده فیلم بر پایه MovieLens 100K</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("MovieMind AI")
    st.caption("کنترل پنل سامانه")
    st.divider()
    st.markdown("**وضعیت مدل**")
    if model is not None:
        st.success("مدل آموزش‌دیده بارگذاری شده")
    else:
        st.warning("مدلی در دسترس نیست")
    if report:
        st.caption(f"آخرین آموزش: {report['created_at']}")
        st.caption(f"زمان آموزش: {report['runtime_seconds']} ثانیه")
    st.divider()
    st.markdown("**پیکربندی فعلی**")
    st.caption("الگوریتم: TruncatedSVD")
    st.caption("تقسیم داده: ۵۰٪ / ۵۰٪")
    st.caption("بذر تصادفی: ۴۲")
    st.caption("دیتاست: MovieLens 100K")
    st.divider()
    if st.button("بارگذاری مجدد مدل ذخیره‌شده", use_container_width=True):
        st.session_state.model = load_saved_model()
        st.rerun()

tab_overview, tab_data, tab_train, tab_recommend, tab_outputs = st.tabs(
    ["نمای کلی", "شناخت داده", "آموزش و ارزیابی", "پیشنهاد شخصی", "گزارش‌ها و خروجی‌ها"]
)

with tab_overview:
    st.subheader("نمای کلی سامانه")
    if report:
        ds = report["dataset"]
        ev = report["evaluation"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("امتیازهای دیتاست", f"{ds['ratings']:,}")
        c2.metric("کاربران", f"{ds['users']:,}")
        c3.metric("فیلم‌ها", f"{ds['movies']:,}")
        c4.metric("میانگین امتیاز", f"{ds['rating_mean']:.2f} / 5")
        st.subheader("آخرین نتیجه ارزیابی")
        a, b, c = st.columns(3)
        a.metric("RMSE", f"{ev['rmse']:.4f}", help="ریشه میانگین مربعات خطا؛ کمتر بهتر است.")
        b.metric("MAE", f"{ev['mae']:.4f}", help="میانگین قدرمطلق خطا؛ کمتر بهتر است.")
        c.metric("نمونه‌های آزمون", f"{ev['test_count']:,}")
        st.caption(
            f"مدل با {report['model']['used_components']} مؤلفه آموزش دیده؛ "
            f"زمان کل فرایند {report['runtime_seconds']} ثانیه بوده است."
        )
    else:
        st.info("هنوز گزارشی از آموزش در این نشست وجود ندارد. برای ساخت گزارش، آموزش مدل را اجرا کنید.")
        c1, c2, c3 = st.columns(3)
        c1.metric("الگوریتم", "SVD")
        c2.metric("نسبت آموزش", "۵۰٪")
        c3.metric("نسبت آزمون", "۵۰٪")

    st.subheader("جریان پردازش")
    flow = [
        ("۱", "دریافت داده", "خواندن رکوردهای امتیازدهی MovieLens"),
        ("۲", "آماده‌سازی", "پاک‌سازی و تقسیم تصادفی با بذر ثابت ۴۲"),
        ("۳", "ساخت ماتریس", "کاربر × فیلم و تکمیل خانه‌های خالی با میانگین کاربر"),
        ("۴", "یادگیری SVD", "استخراج عوامل نهفته و بازسازی امتیازها"),
        ("۵", "ارزیابی", "محاسبه RMSE و MAE روی داده‌های کنارگذاشته‌شده"),
        ("۶", "پیشنهاد", "مرتب‌سازی فیلم‌های امتیازنداده‌شده برای هر کاربر"),
    ]
    for number, heading, description in flow:
        with st.container(border=True):
            x, y = st.columns([1, 8])
            x.markdown(f"### {number}")
            y.markdown(f"**{heading}**")
            y.caption(description)

with tab_data:
    st.subheader("شناخت و کیفیت داده")
    if st.session_state.ratings is None and st.button("بارگذاری دیتاست و نمایش جزئیات", type="primary"):
        try:
            with st.spinner("در حال دریافت یا خواندن MovieLens 100K..."):
                st.session_state.ratings = get_ratings()
        except Exception as exc:
            st.error(f"بارگذاری دیتاست ناموفق بود: {exc}")
    ratings = st.session_state.ratings
    if ratings is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("تعداد رکورد", f"{len(ratings):,}")
        c2.metric("کاربر یکتا", f"{ratings['user_id'].nunique():,}")
        c3.metric("فیلم یکتا", f"{ratings['movie_title'].nunique():,}")
        c4.metric("میانگین امتیاز", f"{ratings['user_rating'].mean():.3f}")
        st.markdown("**توزیع امتیازهای ثبت‌شده**")
        rating_dist = ratings["user_rating"].value_counts().sort_index()
        st.bar_chart(rating_dist.rename_axis("امتیاز").to_frame("تعداد رکورد"))
        left, right = st.columns(2)
        with left:
            st.markdown("**کاربران با بیشترین تعداد امتیاز**")
            active = ratings.groupby("user_id").size().nlargest(10).rename("تعداد امتیاز")
            st.dataframe(active, use_container_width=True)
        with right:
            st.markdown("**فیلم‌های با بیشترین تعداد امتیاز**")
            popular = ratings.groupby("movie_title").size().nlargest(10).rename("تعداد امتیاز")
            st.dataframe(popular, use_container_width=True)
        with st.expander("مشاهده نمونه رکوردهای دیتاست"):
            st.dataframe(ratings.head(30), use_container_width=True)
        st.download_button(
            "دریافت داده‌های بارگذاری‌شده (CSV)",
            ratings.to_csv(index=False).encode("utf-8-sig"),
            "movielens_loaded_ratings.csv",
            "text/csv",
        )
    else:
        st.caption("برای جلوگیری از دانلود ناخواسته، بارگذاری دیتاست با دکمه بالا انجام می‌شود.")

with tab_train:
    st.subheader("مرکز آموزش مدل")
    st.write("پارامترها را انتخاب کنید، فرایند را اجرا کنید و جزئیات هر مرحله را بررسی کنید.")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        components = st.slider("تعداد مؤلفه‌های نهفته SVD", 5, 50, 20, 1)
    with col_b:
        split_ratio = st.selectbox("سهم داده آزمون", ["۵۰٪ (پیشنهادی و ثابت پروژه)"], index=0)
        st.caption("نسبت آموزش نیز ۵۰٪ خواهد بود.")
    with col_c:
        st.metric("بذر تصادفی", "42")
        st.caption("برای تکرارپذیری تقسیم داده")

    with st.expander("SVD چه کاری انجام می‌دهد؟", expanded=False):
        st.markdown(
            "ابتدا ماتریس کاربر–فیلم ساخته می‌شود. خانه‌های خالی با میانگین امتیاز همان کاربر تکمیل می‌شوند. "
            "سپس SVD الگوهای نهفته را استخراج می‌کند و ماتریس امتیازهای تخمینی را بازسازی می‌کند. "
            "این مدل امتیاز احتمالی را تخمین می‌زند؛ خروجی آن احتمال علاقه یا تضمین پسندیدن فیلم نیست."
        )

    if st.button("▶ شروع آموزش کامل", type="primary", use_container_width=True):
        started = perf_counter()
        progress = st.progress(0, text="مرحله ۱ از ۵: بارگذاری دیتاست")
        status = st.empty()
        try:
            status.info("در حال دریافت و آماده‌سازی داده‌های MovieLens 100K...")
            ratings = get_ratings()
            st.session_state.ratings = ratings
            progress.progress(15, text="مرحله ۱ از ۵: دیتاست آماده شد")
            status.success(
                f"{len(ratings):,} امتیاز از {ratings['user_id'].nunique():,} کاربر "
                f"و {ratings['movie_title'].nunique():,} فیلم بارگذاری شد."
            )

            train_data, test_data = split_ratings(ratings, test_size=0.5)
            progress.progress(30, text="مرحله ۲ از ۵: تقسیم آموزش و آزمون")
            status.info(
                f"آموزش: {len(train_data):,} رکورد | آزمون: {len(test_data):,} رکورد"
            )

            status.info("مرحله ۳ از ۵: ساخت ماتریس کاربر–فیلم و آموزش SVD...")
            trained_model = MovieRecommender(n_components=components).fit(train_data)
            progress.progress(70, text="مرحله ۳ از ۵: آموزش مدل کامل شد")

            status.info("مرحله ۴ از ۵: پیش‌بینی و محاسبه معیارهای ارزیابی...")
            metrics = trained_model.evaluate(test_data)
            elapsed_before_save = perf_counter() - started
            report_data, details = build_report(
                ratings, train_data, test_data, trained_model, metrics, elapsed_before_save
            )
            progress.progress(88, text="مرحله ۴ از ۵: ارزیابی کامل شد")

            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            trained_model.save(MODEL_PATH)
            details.to_csv(EVALUATION_PATH, index=False)
            report_data["runtime_seconds"] = round(perf_counter() - started, 3)
            report_data["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            REPORT_PATH.write_text(
                json.dumps(report_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            progress.progress(100, text="مرحله ۵ از ۵: ذخیره مدل و گزارش‌ها کامل شد")

            st.session_state.model = trained_model
            st.session_state.report = report_data
            st.session_state.evaluation_details = details
            st.session_state.recommendations = None
            status.success("آموزش، ارزیابی و ذخیره خروجی‌ها با موفقیت انجام شد.")
            st.rerun()
        except Exception as exc:
            status.error(f"آموزش در یکی از مراحل متوقف شد: {exc}")
            st.exception(exc)

    if st.session_state.report:
        report = st.session_state.report
        st.divider()
        st.subheader("گزارش اجرای آموزش")
        ev = report["evaluation"]
        md = report["model"]
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("RMSE", f"{ev['rmse']:.4f}")
        x2.metric("MAE", f"{ev['mae']:.4f}")
        x3.metric("زمان اجرا", f"{report['runtime_seconds']:.2f} ثانیه")
        x4.metric("واریانس توضیح‌داده‌شده", f"{(md['explained_variance_ratio_sum'] or 0) * 100:.2f}%")
        st.markdown("**جزئیات داده و مدل**")
        details_rows = [
            ("کل امتیازها", f"{report['dataset']['ratings']:,}"),
            ("رکوردهای آموزش", f"{report['split']['train_ratings']:,}"),
            ("رکوردهای آزمون", f"{report['split']['test_ratings']:,}"),
            ("ابعاد ماتریس کاربر–فیلم", f"{md['matrix_users']:,} × {md['matrix_movies']:,}"),
            ("تعداد مؤلفه‌های استفاده‌شده", str(md["used_components"])),
            ("تراکم ماتریس مشاهده‌شده", f"{md['matrix_density'] * 100:.2f}%"),
            ("میانگین سراسری آموزش", f"{md['global_mean']:.4f}"),
            ("پیش‌بینی با میانگین سراسری", f"{ev['fallback_rows']:,} رکورد"),
        ]
        st.dataframe(
            pd.DataFrame(details_rows, columns=["شاخص", "مقدار"]),
            hide_index=True, use_container_width=True,
        )
        st.caption(
            "نکته ارزیابی: رکوردهایی که کاربر یا فیلم آن‌ها در ماتریس آموزش حضور نداشته، "
            "با میانگین سراسری آموزش پیش‌بینی شده‌اند؛ این پیش‌بینی شخصی‌سازی‌شده نیست."
        )

with tab_recommend:
    st.subheader("تولید پیشنهادهای شخصی‌سازی‌شده")
    model = st.session_state.model
    if model is None or model.user_item is None:
        st.warning("ابتدا مدل را در بخش آموزش و ارزیابی آموزش دهید.")
    else:
        users = sorted(model.user_item.index.astype(str).tolist())
        ucol, ncol = st.columns([2, 1])
        with ucol:
            selected_user = st.selectbox("انتخاب شناسه کاربر", users)
        with ncol:
            top_n = st.selectbox("تعداد فیلم پیشنهادی", [5, 10, 15, 20, 30], index=1)
        user_ratings = model.user_item.loc[selected_user].dropna().sort_values(ascending=False)
        p1, p2, p3 = st.columns(3)
        p1.metric("فیلم‌های امتیازداده‌شده در آموزش", f"{len(user_ratings):,}")
        p2.metric("میانگین امتیاز کاربر", f"{user_ratings.mean():.2f}" if len(user_ratings) else "—")
        p3.metric("فیلم‌های قابل پیشنهاد", f"{len(model.predicted_ratings.columns) - len(user_ratings):,}")

        if st.button("ساخت پیشنهادها برای این کاربر", type="primary", use_container_width=True):
            try:
                st.session_state.recommendations = recommend_movies(model, selected_user, top_n)
                st.session_state.recommendation_user = selected_user
            except Exception as exc:
                st.error(f"تولید پیشنهاد ناموفق بود: {exc}")

        result = st.session_state.recommendations
        if result is not None and st.session_state.get("recommendation_user") == selected_user:
            st.markdown("**رتبه‌بندی فیلم‌ها بر اساس امتیاز پیش‌بینی‌شده**")
            shown = result.copy()
            shown["predicted_rating"] = shown["predicted_rating"].round(3)
            shown.index = range(1, len(shown) + 1)
            st.dataframe(
                shown.rename(columns={
                    "user_id": "شناسه کاربر",
                    "movie_title": "عنوان فیلم",
                    "predicted_rating": "امتیاز تخمینی از ۵",
                }),
                use_container_width=True,
            )
            st.bar_chart(result.set_index("movie_title")["predicted_rating"])
            st.caption(
                "امتیاز تخمینی از بازسازی ماتریس SVD به دست آمده است؛ "
                "این مقدار احتمال علاقه‌مندی یا امتیاز قطعی آینده کاربر نیست."
            )
            st.download_button(
                "دانلود پیشنهادهای این کاربر (CSV)",
                result.to_csv(index=False).encode("utf-8-sig"),
                f"recommendations_user_{selected_user}.csv",
                "text/csv",
                use_container_width=True,
            )
        with st.expander("فیلم‌هایی که این کاربر در بخش آموزش امتیاز داده است"):
            st.dataframe(
                user_ratings.rename("امتیاز ثبت‌شده").to_frame(),
                use_container_width=True,
            )

with tab_outputs:
    st.subheader("مرکز گزارش‌ها و فایل‌های خروجی")
    st.write("فایل‌های زیر پس از اجرای موفق آموزش در پوشه artifacts ذخیره می‌شوند.")
    if report:
        st.success("گزارش آموزش در این نشست موجود است.")
        report_json = json.dumps(report, ensure_ascii=False, indent=2)
        st.download_button(
            "دریافت گزارش کامل آموزش (JSON)",
            report_json,
            "training_report.json",
            "application/json",
            use_container_width=True,
        )
        details = st.session_state.evaluation_details
        if details is not None:
            st.markdown("**جزئیات پیش‌بینی تک‌تک رکوردهای آزمون**")
            st.caption("شامل امتیاز واقعی، امتیاز پیش‌بینی‌شده، خطای مطلق و منبع پیش‌بینی.")
            st.download_button(
                "دریافت جزئیات ارزیابی (CSV)",
                details.to_csv(index=False).encode("utf-8-sig"),
                "evaluation_predictions.csv",
                "text/csv",
                use_container_width=True,
            )
            with st.expander("پیش‌نمایش ۵۰ پیش‌بینی آزمون"):
                preview = details.head(50).copy()
                for col in ["actual_rating", "predicted_rating", "absolute_error"]:
                    preview[col] = preview[col].round(3)
                st.dataframe(preview, use_container_width=True)
            st.markdown("**توزیع خطای مطلق در داده آزمون**")
            bins = pd.cut(
                details["absolute_error"],
                bins=[-0.001, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0],
                labels=["۰ تا ۰٫۲۵", "۰٫۲۵ تا ۰٫۵", "۰٫۵ تا ۱", "۱ تا ۱٫۵", "۱٫۵ تا ۲", "بیشتر از ۲"],
            )
            st.bar_chart(bins.value_counts(sort=False).rename("تعداد رکورد"))
        st.markdown("**فایل‌های ذخیره‌شده روی سیستم**")
        for path in [MODEL_PATH, REPORT_PATH, EVALUATION_PATH]:
            st.write(f"{'✅' if path.exists() else '—'}  `{path}`")
    else:
        st.info("هنوز خروجی آموزشی تولید نشده است. ابتدا آموزش مدل را اجرا کنید.")

st.divider()
st.caption("MovieMind AI • SVD Collaborative Filtering • MovieLens 100K")
