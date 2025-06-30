import streamlit as st  # استيراد مكتبة Streamlit لإنشاء واجهة الويب
import yfinance as yf  # استيراد yfinance لتحميل بيانات البورصة مثل البيتكوين
import pandas as pd  # استيراد pandas لمعالجة البيانات
# from sklearn.linear_model import LinearRegression  # استيراد نموذج الانحدار الخطي من scikit-learn
from sklearn.ensemble import RandomForestRegressor  # ✅ استيراد نموذج غير خطي بديل (Random Forest)
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error  # ✅ استيراد أدوات التقييم
import matplotlib.pyplot as plt  # استيراد مكتبة matplotlib للرسم البياني
from datetime import date, timedelta  # استيراد التاريخ واليوم والإضافة الزمنية

# Başlık
st.title("🔮 Bitcoin Fiyat Tahmini")  # عنوان التطبيق في الصفحة

# Veri yükleme
st.subheader("📈 Bitcoin Fiyat Verileri")  # عنوان فرعي لقسم تحميل البيانات
start_date = "2024-01-01"  # ✅ تاريخ بداية البيانات (تم التعديل ليكون 2024 فقط)
end_date = "2024-12-31"  # ✅ تاريخ نهاية البيانات (تم التعديل ليكون 2024 فقط)

# Verileri indir
df = yf.download("BTC-USD", start=start_date, end=end_date)  # تحميل بيانات البيتكوين من Yahoo Finance

# Veri boşsa uyarı göster
if df.empty:  # إذا لم يتم تحميل أي بيانات
    st.error("❌ Bitcoin verileri alınamadı. İnternet bağlantınızı veya tarih aralığını kontrol edin.")  # عرض رسالة خطأ
    st.stop()  # إيقاف تنفيذ التطبيق

# Grafik: Kapanış fiyatı
st.line_chart(df['Close'])  # رسم مخطط خطي لأسعار الإغلاق اليومية

# Basit tahmin modeli
st.subheader("🤖 Tahmin Modeli (Basit)")  # عنوان فرعي لقسم التوقعات

df = df.reset_index()  # إعادة ترتيب الفهرسة لجعل التاريخ عمود عادي
df['Days'] = (df['Date'] - pd.to_datetime(start_date)).dt.days  # تحويل التواريخ إلى عدد الأيام منذ بداية الفترة

# ✅ استخدام أول 20 يوم فقط من يناير للتدريب
train_data = df[df['Date'] <= '2024-01-20']  # تصفية أول 20 يوم فقط
X_train = train_data[['Days']]  # المدخلات (عدد الأيام لأول 20 يوم)
y_train = train_data['Close']  # الإخراجات (أسعار الإغلاق لأول 20 يوم)

# Modeli eğit
# model = LinearRegression()  # إنشاء نموذج انحدار خطي
model = RandomForestRegressor(n_estimators=100, random_state=42)  # ✅ استخدام نموذج غير خطي (Random Forest)
model.fit(X_train, y_train)  # تدريب النموذج على أول 20 يوم فقط

# ✅ توقع سعر اليوم 21 يناير 2024
target_date = pd.to_datetime("2024-01-21")
target_day = (target_date - pd.to_datetime(start_date)).days
predicted_price = model.predict([[target_day]])  # التوقع لليوم 21

# Tahmini göster
st.success(f"📅 السعر المتوقع ليوم 21 يناير 2024: **{float(predicted_price[0]):.2f}$**")  # عرض التوقع النهائي

# ✅ السعر الحقيقي في اليوم 21 (لو موجود)
real_price = df[df['Date'] == target_date]['Close']
if not real_price.empty:
    real_value = float(real_price.values[0])
    st.info(f"📊 السعر الحقيقي في 21 يناير 2024: **{real_value:.2f}$**")

    # ✅ تقييم النموذج باستخدام MAE و MAPE
    mae = mean_absolute_error([real_value], predicted_price)
    mape = mean_absolute_percentage_error([real_value], predicted_price)

    st.subheader("📉 تقييم أداء النموذج")
    st.write(f"📐 متوسط الخطأ المطلق (MAE): **{mae:.2f}**")
    st.write(f"📐 متوسط نسبة الخطأ المطلق (MAPE): **{mape * 100:.2f}%**")
else:
    st.warning("⚠️ لا توجد بيانات حقيقية لليوم 21 يناير 2024 لتقييم النموذج.")

# Grafik gösterimi
st.subheader("📊 Gerçek Veriler vs Tahmin")  # عنوان فرعي لمقارنة التوقع بالواقع
plt.figure(figsize=(10, 4))  # تحديد حجم الشكل البياني
plt.plot(df['Date'], df['Close'], label="Gerçek Fiyat")  # رسم خط يمثل الأسعار الحقيقية
plt.scatter(target_date, predicted_price, color='red', label="Tahmin (21 Jan)")  # وضع نقطة توقع السعر
plt.xlabel("Tarih")  # عنوان المحور السيني (التاريخ)
plt.ylabel("Fiyat")  # عنوان المحور الصادي (السعر)
plt.legend()  # عرض وسيلة الإيضاح
st.pyplot(plt)  # عرض الرسم البياني داخل واجهة Streamlit
