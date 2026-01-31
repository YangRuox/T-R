# convert_to_json_fixed.py
import json
import numpy as np
import joblib
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def load_multilingual_jobs():
    """加载所有语言的职业名称"""
    print("Loading multilingual job files...")
    
    languages = ['en', 'zh', 'es', 'fr', 'ru', 'ar']
    job_translations = {}
    
    for lang in languages:
        filepath = f"job_{lang}.npy"
        try:
            data = np.load(filepath, allow_pickle=True)
            job_list = data.tolist()
            job_translations[lang] = job_list
            print(f"  ✅ {filepath} - {len(job_list)} jobs")
            
            # 显示示例
            if len(job_list) > 0:
                print(f"    示例: {job_list[:3]}")
        except FileNotFoundError:
            print(f"  ❌ {filepath} not found")
            job_translations[lang] = []
    
    # 验证所有语言文件长度一致
    lengths = [len(job_translations[lang]) for lang in languages if lang in job_translations]
    if lengths and len(set(lengths)) > 1:
        print(f"⚠️ 警告: 语言文件长度不一致: {lengths}")
    
    return job_translations

def load_base_job_data():
    """加载基础职业数据"""
    print("\nLoading base job data...")
    
    base_data = {}
    
    # 加载职业代码
    try:
        job_codes = np.load("job_codes.npy", allow_pickle=True).tolist()
        base_data['job_codes'] = job_codes
        print(f"  ✅ job_codes.npy - {len(job_codes)} codes")
        print(f"    示例: {job_codes[:5]}")
    except FileNotFoundError:
        print("  ❌ job_codes.npy not found")
        base_data['job_codes'] = []
    
    # 加载PCA权重
    try:
        pca_weights = np.load("pca_weights.npy", allow_pickle=True).tolist()
        base_data['pca_weights'] = pca_weights
        print(f"  ✅ pca_weights.npy - shape: {np.array(pca_weights).shape if pca_weights else 'N/A'}")
    except FileNotFoundError:
        print("  ❌ pca_weights.npy not found")
        base_data['pca_weights'] = []
    
    # 加载职业特征
    try:
        scaled_features = np.load("scaled_job_features.npy", allow_pickle=True).tolist()
        base_data['scaled_features'] = scaled_features
        print(f"  ✅ scaled_job_features.npy - {len(scaled_features)} jobs")
    except FileNotFoundError:
        print("  ❌ scaled_job_features.npy not found")
        base_data['scaled_features'] = []
    
    return base_data

def load_questions():
    """加载问题数据"""
    print("\nLoading questions.tsv...")
    
    languages = ['en', 'zh', 'es', 'fr', 'ru', 'ar']
    questions_dict = {}
    
    try:
        questions_df = pd.read_csv('questions.tsv', sep='\t')
        print(f"  ✅ Loaded questions.tsv - {len(questions_df)} questions")
        
        for lang in languages:
            if lang in questions_df.columns:
                questions = questions_df[lang].fillna('').astype(str).tolist()
                questions_dict[lang] = questions
                print(f"    ✅ {lang}: {len(questions)} questions")
            else:
                print(f"    ⚠️ {lang} not found, using English")
                questions_dict[lang] = questions_df['en'].fillna('').astype(str).tolist()
        
        return questions_dict, len(questions_df)
    
    except FileNotFoundError:
        print("❌ questions.tsv not found")
        return {}, 0

def convert_tsv_files():
    """将所有的.tsv文件转换为JSON格式"""
    print("\nConverting .tsv files...")
    
    tsv_data = {}
    
    try:
        # 加载meanNorms.tsv
        mean_norms = pd.read_csv('meanNorms.tsv', sep='\t')
        mean_norms_json = {}
        for idx, row in mean_norms.iterrows():
            group = int(row['group'])
            values = row[1:].tolist()
            mean_norms_json[group] = values
        tsv_data['mean_norms'] = mean_norms_json
        print(f"  ✅ meanNorms.tsv - {len(mean_norms_json)} groups")
    except FileNotFoundError:
        print("  ❌ meanNorms.tsv not found")
        tsv_data['mean_norms'] = {}
    
    try:
        # 加载sdNorms.tsv
        sd_norms = pd.read_csv('sdNorms.tsv', sep='\t')
        sd_norms_json = {}
        for idx, row in sd_norms.iterrows():
            group = int(row['group'])
            values = row[1:].tolist()
            sd_norms_json[group] = values
        tsv_data['sd_norms'] = sd_norms_json
        print(f"  ✅ sdNorms.tsv - {len(sd_norms_json)} groups")
    except FileNotFoundError:
        print("  ❌ sdNorms.tsv not found")
        tsv_data['sd_norms'] = {}
    
    try:
        # 加载weightsB5.tsv
        weights = pd.read_csv('weightsB5.tsv', sep='\t')
        tsv_data['weights'] = weights.values.tolist()
        print(f"  ✅ weightsB5.tsv - {len(tsv_data['weights'])} questions × 5 traits")
    except FileNotFoundError:
        print("  ❌ weightsB5.tsv not found")
        tsv_data['weights'] = []
    
    return tsv_data

def load_scaler():
    """加载和转换scaler"""
    print("\nLoading scaler...")
    
    try:
        scaler = joblib.load("your_scaler.pkl")
        
        scaler_params = {
            "mean": scaler.mean_.tolist() if hasattr(scaler, 'mean_') else [],
            "scale": scaler.scale_.tolist() if hasattr(scaler, 'scale_') else [],
            "var": scaler.var_.tolist() if hasattr(scaler, 'var_') else [],
            "n_samples_seen": int(scaler.n_samples_seen_) if hasattr(scaler, 'n_samples_seen_') else 0,
            "feature_names": ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"]
        }
        
        print("  ✅ your_scaler.pkl loaded")
        return scaler_params
    
    except FileNotFoundError:
        print("  ❌ your_scaler.pkl not found")
        return {}

def load_other_files():
    """加载其他支持文件"""
    print("\nLoading other support files...")
    
    other_data = {}
    
    try:
        text_dict = np.load("text_dict.npy", allow_pickle=True).item()
        other_data['text_dict'] = text_dict
        print("  ✅ text_dict.npy loaded")
    except:
        print("  ❌ text_dict.npy not found")
        other_data['text_dict'] = {}
    
    try:
        language_display = np.load("language_display.npy", allow_pickle=True).item()
        other_data['language_display'] = language_display
        print("  ✅ language_display.npy loaded")
    except:
        print("  ❌ language_display.npy not found")
        other_data['language_display'] = {}
    
    return other_data

def get_translation_texts():
    """获取所有翻译文本"""
    return {
        "titles": {
            "en": "Your Big Five Personality Profile (T scores)",
            "zh": "你的大五人格雷达图（T分）",
            "es": "Tu Perfil de Personalidad Big Five (Puntajes T)",
            "fr": "Votre profil de personnalité Big Five (scores T)",
            "ru": "Ваш профиль личности по Big Five (T-баллы)",
            "ar": "ملف الشخصية الخاص بك (Big Five) بدرجات T",
        },
        "trait_names": {
            "en": ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"],
            "zh": ["神经质", "外向性", "开放性", "宜人性", "尽责性"],
            "es": ["Neuroticismo", "Extraversión", "Apertura", "Amabilidad", "Conciencia"],
            "fr": ["Névrosisme", "Extraversion", "Ouverture", "Amabilité", "Conscienciosité"],
            "ru": ["Невротизм", "Экстраверсия", "Открытость", "Доброжелательность", "Сознательность"],
            "ar": ["الضيق العصبي", "الانفتاح", "الانبساطية", "التعاطف", "الضمير المهني"]
        },
        "disclaimer": {
            "en": "The career recommendations provided here are for your reference only. It's important to consider your personal circumstances, preferences, and goals when making a career decision. We encourage you to explore different options and take the time to evaluate each one carefully. May you find a fulfilling and rewarding career that aligns with your values and aspirations. Best of luck on your journey to success! 😊",
            "zh": "这里提供的职业推荐仅供参考。在做出职业选择时，请务必考虑您的个人情况、兴趣和目标。我们鼓励您探索不同的职业选项，并仔细评估每一个选择。希望您能够找到一个符合自己价值观和人生目标的理想职业，祝您在职业生涯中取得圆满成功！😊",
            "es": "Las recomendaciones de carrera proporcionadas aquí son solo para su referencia. Es importante tener en cuenta sus circunstancias personales, preferencias y objetivos al tomar una decisión sobre su carrera. Le animamos a explorar diferentes opciones y tomarse el tiempo necesario para evaluar cada una de ellas con cuidado. ¡Le deseamos mucho éxito en su camino hacia una carrera gratificante y satisfactoria! 😊",
            "fr": "Les recomendaciones de carrière fournies aquí son uniquement à titre de référence. Il est important de prendre en compte vos circonstances personnelles, vos préférences et vos objectifs lorsque vous prenez una décision concernant votre carrière. Nous vous encourageons à explorer diferentes opciones et à prendre le temps d'évaluer cada choix avec soin. Nous vous souhaitons de trouver une carrière épanouissante et gratifiante qui corresponde à vos valeurs et aspirations. Bonne chance dans votre parcours vers le succès ! 😊",
            "ru": "Предоставленные рекомендации по карьере предназначены только для вашего ознакомления. Важно учитывать ваши личные обстоятельства, предпочтения и цели при принятии решения о карьере. Мы призываем вас исследовать различные варианты и уделять достаточно времени на тщательную оценку каждого из них. Желаем вам найти карьеру, которая будет соответствовать вашим ценностям и устремлениям, и успешного пути к успеху! 😊",
            "ar": "التوصيات المهنية المقدمة здесь предназначены только для справки. Важно учитывать ваши личные обстоятельства, предпочтения и цели при принятии решения о карьере. Мы призываем вас исследовать различные варианты и уделять достаточно времени на тщательную оценку каждого из них. Желаем вам найти карьеру, которая будет соответствовать вашим ценностям и устремлениям, и успешного пути к успеху! 😊"
        },
        "ideal_job_prompt": {
            "en": "Please enter your ideal career (e.g., Data Scientist):",
            "zh": "请输入您的理想职业（例如：数据科学家）：",
            "es": "Por favor, introduzca su carrera ideal (por ejemplo: Científico de datos):",
            "fr": "Veuillez saisir votre métier idéal (par exemple : Data Scientist) :",
            "ru": "Пожалуйста, введите вашу идеальную профессию (например: специалист по данным):",
            "ar": "يرجى إدخال مهنتك المثالية (مثال: عالم بيانات):"
        },
        "ideal_job_warning": {
            "en": "⚠️ Please enter your ideal career.",
            "zh": "⚠️ 请输入您的理想职业。",
            "es": "⚠️ Por favor, introduzca su carrera ideal.",
            "fr": "⚠️ Veuillez saisir votre métier idéal.",
            "ru": "⚠️ Пожалуйста, введите вашу идеальную профессию.",
            "ar": "⚠️ يرجى إدخال مهنتك المثالية."
        },
        "ideal_job_result": {
            "en": "The career closest to your ideal is: **{}**",
            "zh": "您的理想职业最相近的是：**{}**",
            "es": "La carrera más cercana a su ideal es: **{}**",
            "fr": "Le métier le plus proche de votre idéal est : **{}**",
            "ru": "Самая близкая к вашей идеальной профессия: **{}**",
            "ar": "أقرب مهنة к вашей идеальной профессии: **{}**"
        },
        "closest_text": {
            "en": "Your trait closest to the ideal career:",
            "zh": "与理想职业特征最接近的是：", 
            "es": "Tu rasgo más cercano al trabajo ideal:",
            "fr": "Votre trait le plus proche du métier idéal :",
            "ru": "Ваша черта, наиболее близкая к идеальной профессии:",
            "ar": "سمتك الأقرب إلى المهنة المثالية:"
        },
        "furthest_text": {
            "en": "Your trait furthest from the ideal career:",
            "zh": "与理想职业特征差距最大的是：",
            "es": "Tu rasgo más alejado del trabajo ideal:",
            "fr": "Votre trait le plus éloigné du métier idéal :", 
            "ru": "Ваша черта, наиболее далёкая от идеальной профессии:",
            "ar": "سمتك الأبعد عن المهنة المثالية:"
        }
    }

def get_metadata():
    """获取元数据"""
    return {
        "version": "2.0.0",
        "created_at": pd.Timestamp.now().isoformat(),
        "data_sources": [
            "job_en.npy, job_zh.npy, job_es.npy, job_fr.npy, job_ru.npy, job_ar.npy",
            "job_codes.npy",
            "scaled_job_features.npy",
            "pca_weights.npy",
            "meanNorms.tsv",
            "sdNorms.tsv",
            "questions.tsv",
            "weightsB5.tsv",
            "your_scaler.pkl"
        ],
        "languages_supported": ['en', 'zh', 'es', 'fr', 'ru', 'ar'],
        "n_jobs": 263,
        "model_info": {
            "name": "JobRecommenderMLP",
            "input_dim": 5,
            "hidden_dim": 128,
            "output_dim": 263
        }
    }

def convert_all_data():
    """主函数：转换所有数据到JSON格式"""
    print("=" * 60)
    print("Converting all data to JSON format...")
    print("=" * 60)
    
    try:
        # 1. 加载多语言职业名称
        job_translations = load_multilingual_jobs()
        
        # 2. 加载基础职业数据
        base_data = load_base_job_data()
        
        # 3. 加载其他文件
        other_data = load_other_files()
        
        # 4. 加载问题数据
        questions_dict, n_questions = load_questions()
        
        # 5. 加载TSV数据
        tsv_data = convert_tsv_files()
        
        # 6. 加载Scaler
        scaler_params = load_scaler()
        
        # 7. 获取翻译文本
        translations = get_translation_texts()
        
        # 8. 构建完整的数据结构
        print("\n" + "=" * 60)
        print("Building complete data structure...")
        print("=" * 60)
        
        complete_data = {
            # 职业数据 - 关键部分
            "job_translations": job_translations,  # 多语言职业名称
            "job_codes": base_data.get('job_codes', []),
            "pca_weights": base_data.get('pca_weights', []),
            "scaled_features": base_data.get('scaled_features', []),
            
            # 向后兼容：使用英文作为默认职业名称
            "job_names": job_translations.get('en', []),
            
            # 其他数据
            **other_data,
            
            # 问题数据
            "questions": questions_dict,
            
            # 标准化数据
            **tsv_data,
            
            # Scaler参数
            "scaler_params": scaler_params,
            
            # 翻译文本
            "translations": translations,
            
            # 元数据
            "metadata": {
                **get_metadata(),
                "n_jobs": len(job_translations.get('en', [])),
                "n_questions": n_questions,
                "languages_available": list(job_translations.keys())
            },
            
            # 分组信息
            "norm_groups": {
                1: {"gender": "Female", "age_range": "<35", "description": "Female under 35"},
                2: {"gender": "Female", "age_range": ">=35", "description": "Female 35 and over"},
                3: {"gender": "Male", "age_range": "<35", "description": "Male under 35"},
                4: {"gender": "Male", "age_range": ">=35", "description": "Male 35 and over"}
            }
        }
        
        # 9. 保存为JSON文件
        print("\nSaving to JSON file...")
        output_file = "app_data_complete.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(complete_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Successfully saved to {output_file}")
        print(f"📁 File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
        
        # 10. 打印摘要
        print("\n" + "=" * 60)
        print("📊 Data Summary:")
        print("=" * 60)
        
        n_jobs = len(complete_data.get('job_translations', {}).get('en', []))
        print(f"   • Total Jobs: {n_jobs}")
        print(f"   • Total Questions: {n_questions}")
        print(f"   • Languages Available: {len(complete_data.get('job_translations', {}))}")
        print(f"   • Job Features: {len(complete_data.get('scaled_features', []))}")
        print(f"   • Norm Groups: 4")
        print(f"   • Question Languages: {list(questions_dict.keys())}")
        
        # 验证数据完整性
        print("\n🔍 Data Validation:")
        print(f"   • Job translations: {all(len(v) == n_jobs for v in complete_data.get('job_translations', {}).values())}")
        print(f"   • Job codes: {len(complete_data.get('job_codes', [])) == n_jobs}")
        print(f"   • Scaled features: {len(complete_data.get('scaled_features', [])) == n_jobs}")
        
        # 显示示例
        print("\n👀 Example Data (first 3 jobs in each language):")
        for lang in ['en', 'zh', 'es', 'fr', 'ru', 'ar']:
            if lang in complete_data.get('job_translations', {}):
                jobs = complete_data['job_translations'][lang][:3]
                print(f"   • {lang.upper()}: {jobs}")
        
        return output_file
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_compact_version():
    """创建压缩版本，减少文件大小"""
    print("\n" + "=" * 60)
    print("Creating compact version...")
    print("=" * 60)
    
    try:
        with open("app_data_complete.json", "r", encoding="utf-8") as f:
            full_data = json.load(f)
        
        # 创建压缩版本，移除不需要的字段
        compact_data = {
            "job_translations": full_data.get("job_translations", {}),
            "job_codes": full_data.get("job_codes", []),
            "scaled_features": full_data.get("scaled_features", []),
            "pca_weights": full_data.get("pca_weights", []),
            "questions": full_data.get("questions", {}),
            "mean_norms": full_data.get("mean_norms", {}),
            "sd_norms": full_data.get("sd_norms", {}),
            "weights": full_data.get("weights", []),
            "scaler_params": full_data.get("scaler_params", {}),
            "translations": {
                "trait_names": full_data.get("translations", {}).get("trait_names", {}),
                "disclaimer": full_data.get("translations", {}).get("disclaimer", {})
            },
            "metadata": {
                "n_jobs": len(full_data.get("job_translations", {}).get("en", [])),
                "n_questions": len(full_data.get("questions", {}).get("en", [])),
                "languages": list(full_data.get("job_translations", {}).keys())
            }
        }
        
        with open("app_data_compact.json", "w", encoding="utf-8") as f:
            json.dump(compact_data, f, ensure_ascii=False, separators=(',', ':'))
        
        print(f"✅ Compact version saved to app_data_compact.json")
        print(f"📁 File size: {os.path.getsize('app_data_compact.json') / 1024:.2f} KB")
        
    except Exception as e:
        print(f"❌ Failed to create compact version: {e}")

if __name__ == "__main__":
    # 转换完整数据
    output_file = convert_all_data()
    
    if output_file:
        # 创建压缩版本
        create_compact_version()
        
        print("\n" + "=" * 60)
        print("🎉 All conversions complete!")
        print("=" * 60)
        
        print("\n📂 Generated files:")
        print("  • app_data_complete.json - Complete dataset with all translations")
        print("  • app_data_compact.json  - Compact version (smaller file size)")
        
        print("\n🚀 Next steps:")
        print("1. Update your HTML to load the correct JSON file:")
        print("""
   // In your HTML/JS, change the loading code:
   fetch('app_data_complete.json')  // or 'app_data_compact.json'
     .then(response => response.json())
     .then(data => {
       appData = data;
       console.log('Loaded', data.metadata.n_jobs, 'jobs in', 
                   data.metadata.languages.length, 'languages');
     });
        """)
        
        print("\n2. Update your JavaScript to use multilingual job names:")
        print("""
   // When displaying job recommendations:
   function getJobName(jobIndex, language) {
     if (appData.job_translations && appData.job_translations[language]) {
       return appData.job_translations[language][jobIndex];
     }
     return appData.job_names[jobIndex]; // fallback
   }
        """)
    else:
        print("\n❌ Conversion failed. Please check the error messages above.")