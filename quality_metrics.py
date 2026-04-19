import numpy as np
from skimage.metrics import structural_similarity as ssim

class QualityMetrics:
    """Класс для расчёта метрик качества восстановления"""
    
    @staticmethod
    def rmse(original, recovered):
        """Среднеквадратичная ошибка (в единицах изображения)"""
        return np.sqrt(np.mean((recovered - original) ** 2))
    
    @staticmethod
    def rmse_percent(original, recovered):
        """Относительная RMSE в процентах от диапазона яркости"""
        rmse = np.sqrt(np.mean((recovered - original) ** 2))
        dynamic_range = original.max() - original.min()
        return (rmse / dynamic_range) * 100 if dynamic_range != 0 else 0.0
    
    @staticmethod
    def ssim(original, recovered, data_range=None):
        """
        Структурное сходство (Structural Similarity Index)
        
        Параметры:
            original: оригинальное изображение
            recovered: восстановленное изображение
            data_range: динамический диапазон (если None, вычисляется автоматически)
        
        Возвращает:
            ssim_value: float от 0 до 1 (1 = идеальное сходство)
        """
        if data_range is None:
            data_range = original.max() - original.min()
        
        # skimage ожидает изображения в диапазоне [0, data_range]
        # убедимся, что значения не выходят за пределы
        orig_clipped = np.clip(original, 0, data_range)
        rec_clipped = np.clip(recovered, 0, data_range)
        
        return ssim(orig_clipped, rec_clipped, data_range=data_range)
