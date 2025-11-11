import logging
from datetime import datetime
from typing import List, Dict, Any
import random
import json

logger = logging.getLogger(__name__)

class DynamicRecommendationEngine:
    """
    Dynamic Recommendation Engine for Workplace Stress Management
    
    This engine provides personalized recommendations based on:
    - Overall stress level
    - Individual stress factors (workload, work-life balance, team conflict, etc.)
    - Department-specific considerations
    - Severity thresholds
    """
    
    def __init__(self):
        self.recommendations = {
            'workload_management': [
                {
                    'id': 'wm_01',
                    'title': 'Atur Prioritas dan Delegasikan Tugas',
                    'description': 'Fokus pada pengelolaan beban kerja secara efisien. Prioritaskan tugas-tugas penting dan delegasikan jika memungkinkan untuk mengurangi beban berlebih.',
                    'category': 'workload',
                    'priority': 'high',
                    'implementation_steps': [
                        'Buat matriks prioritas untuk semua tugas saat ini',
                        'Identifikasi tugas yang dapat didelegasikan kepada anggota tim',
                        'Tetapkan tenggat waktu yang realistis dan komunikasikan dengan jelas',
                        'Gunakan teknik time-blocking untuk kerja yang terfokus'
                    ]
                },
                {
                    'id': 'wm_02', 
                    'title': 'Pelatihan Manajemen Waktu',
                    'description': 'Ikuti pelatihan manajemen waktu untuk mempelajari strategi menangani beban kerja tinggi dan tenggat waktu secara efektif.',
                    'category': 'workload',
                    'priority': 'medium',
                    'implementation_steps': [
                        'Daftarkan diri dalam workshop manajemen waktu',
                        'Pelajari dan terapkan Teknik Pomodoro',
                        'Gunakan alat digital untuk pelacakan tugas dan penjadwalan',
                        'Latihan mengatakan tidak pada komitmen yang tidak penting'
                    ]
                },
                {
                    'id': 'wm_03',
                    'title': 'Strategi Redistribusi Beban Kerja',
                    'description': 'Terapkan pendekatan sistematis untuk mendistribusikan kembali beban kerja berlebih di antara anggota tim untuk keseimbangan yang lebih baik.',
                    'category': 'workload',
                    'priority': 'high',
                    'implementation_steps': [
                        'Lakukan penilaian beban kerja di seluruh tim',
                        'Identifikasi anggota tim yang belum dimanfaatkan secara optimal',
                        'Buat sistem distribusi tugas yang adil',
                        'Pantau keseimbangan beban kerja secara berkala'
                    ]
                }
            ],
            'work_life_balance': [
                {
                    'id': 'wlb_01',
                    'title': 'Tetapkan Batasan yang Sehat',
                    'description': 'Pastikan keseimbangan kerja-hidup yang tepat dengan menetapkan batasan yang jelas antara waktu kerja dan waktu pribadi.',
                    'category': 'work_life_balance',
                    'priority': 'high',
                    'implementation_steps': [
                        'Tetapkan jam kerja yang spesifik dan patuhi',
                        'Matikan notifikasi kerja setelah jam kerja',
                        'Buat ruang kerja khusus yang terpisah dari area tempat tinggal',
                        'Jadwalkan waktu pribadi dan aktivitas secara rutin'
                    ]
                },
                {
                    'id': 'wlb_02',
                    'title': 'Pengaturan Kerja Fleksibel',
                    'description': 'Terapkan jam kerja fleksibel atau opsi kerja jarak jauh untuk meningkatkan keseimbangan kerja-hidup dan mengurangi stres perjalanan.',
                    'category': 'work_life_balance',
                    'priority': 'medium',
                    'implementation_steps': [
                        'Diskusikan opsi jadwal fleksibel dengan manajemen',
                        'Usulkan pengaturan kerja hybrid',
                        'Tetapkan protokol komunikasi yang jelas untuk kerja jarak jauh',
                        'Siapkan ruang kantor rumah yang ergonomis'
                    ]
                },
                {
                    'id': 'wlb_03',
                    'title': 'Kebijakan Istirahat dan Cuti Reguler',
                    'description': 'Dorong pengambilan istirahat teratur dan penggunaan waktu cuti untuk mencegah burnout dan menjaga kesehatan mental.',
                    'category': 'work_life_balance',
                    'priority': 'medium',
                    'implementation_steps': [
                        'Jadwalkan istirahat singkat setiap 2 jam selama kerja',
                        'Rencanakan dan ambil cuti tahunan secara teratur',
                        'Praktikkan micro-break (5-10 menit) sepanjang hari',
                        'Lakukan aktivitas non-kerja selama istirahat'
                    ]
                }
            ],
            'team_communication': [
                {
                    'id': 'tc_01',
                    'title': 'Tingkatkan Komunikasi Tim',
                    'description': 'Perkuat komunikasi tim dengan menetapkan ekspektasi yang jelas dan check-in rutin untuk mengurangi kesalahpahaman.',
                    'category': 'team_conflict',
                    'priority': 'high',
                    'implementation_steps': [
                        'Terapkan rapat tim mingguan dengan agenda yang jelas',
                        'Tetapkan protokol dan saluran komunikasi',
                        'Praktikkan teknik mendengarkan aktif',
                        'Buat mekanisme umpan balik untuk perbaikan berkelanjutan'
                    ]
                },
                {
                    'id': 'tc_02',
                    'title': 'Pelatihan Resolusi Konflik',
                    'description': 'Berikan pelatihan resolusi konflik kepada anggota tim untuk menangani perselisihan di tempat kerja secara konstruktif.',
                    'category': 'team_conflict',
                    'priority': 'high',
                    'implementation_steps': [
                        'Adakan workshop resolusi konflik',
                        'Tetapkan prosedur mediasi',
                        'Latih manajer dalam manajemen konflik',
                        'Buat saluran pelaporan anonim untuk masalah'
                    ]
                },
                {
                    'id': 'tc_03',
                    'title': 'Aktivitas Membangun Tim',
                    'description': 'Adakan aktivitas team building secara rutin untuk memperkuat hubungan dan meningkatkan kolaborasi.',
                    'category': 'team_conflict',
                    'priority': 'medium',
                    'implementation_steps': [
                        'Rencanakan latihan team building bulanan',
                        'Adakan pertemuan sosial informal',
                        'Buat tim proyek lintas fungsi',
                        'Terapkan program pengakuan rekan kerja'
                    ]
                }
            ],
            'stress_management': [
                {
                    'id': 'sm_01',
                    'title': 'Program Mindfulness dan Meditasi',
                    'description': 'Terapkan program mindfulness dan meditasi untuk membantu karyawan mengelola stres dan meningkatkan fokus.',
                    'category': 'general',
                    'priority': 'high',
                    'implementation_steps': [
                        'Perkenalkan sesi meditasi harian 10 menit',
                        'Berikan akses ke aplikasi meditasi atau sumber daya',
                        'Buat ruang tenang untuk praktik mindfulness',
                        'Tawarkan workshop meditasi terpandu'
                    ]
                },
                {
                    'id': 'sm_02',
                    'title': 'Program Bantuan Karyawan',
                    'description': 'Berikan program bantuan karyawan yang komprehensif termasuk konseling dan dukungan kesehatan mental.',
                    'category': 'general',
                    'priority': 'high',
                    'implementation_steps': [
                        'Bermitra dengan profesional kesehatan mental',
                        'Tawarkan layanan konseling yang rahasia',
                        'Berikan sumber daya dan alat manajemen stres',
                        'Buat kampanye kesadaran kesehatan mental'
                    ]
                },
                {
                    'id': 'sm_03',
                    'title': 'Program Kesehatan Fisik',
                    'description': 'Promosikan kesehatan fisik melalui program olahraga, perbaikan ergonomis, dan inisiatif kesehatan.',
                    'category': 'general',
                    'priority': 'medium',
                    'implementation_steps': [
                        'Sediakan fasilitas fitness di tempat atau keanggotaan gym',
                        'Lakukan penilaian ergonomis pada workstation',
                        'Adakan rapat berjalan dan meja berdiri',
                        'Tawarkan pemeriksaan kesehatan dan wellness check'
                    ]
                }
            ],
            'management_support': [
                {
                    'id': 'ms_01',
                    'title': 'Pelatihan Kepemimpinan untuk Manajer',
                    'description': 'Berikan pelatihan kepemimpinan untuk membantu manajer lebih baik mendukung tim mereka dan mengenali indikator stres.',
                    'category': 'management_support',
                    'priority': 'high',
                    'implementation_steps': [
                        'Latih manajer tentang kecerdasan emosional',
                        'Ajarkan teknik pengenalan dan intervensi stres',
                        'Terapkan check-in satu-satu secara rutin',
                        'Berikan sumber daya untuk mendukung karyawan yang kesulitan'
                    ]
                },
                {
                    'id': 'ms_02',
                    'title': 'Implementasi Kebijakan Pintu Terbuka',
                    'description': 'Tetapkan kebijakan pintu terbuka di mana karyawan merasa nyaman mendiskusikan kekhawatiran dengan manajemen.',
                    'category': 'management_support',
                    'priority': 'medium',
                    'implementation_steps': [
                        'Komunikasikan kebijakan pintu terbuka dengan jelas',
                        'Latih manajer tentang komunikasi yang dapat didekati',
                        'Buat sesi umpan balik yang terstruktur',
                        'Tindak lanjuti kekhawatiran karyawan dengan cepat'
                    ]
                }
            ],
            'work_environment': [
                {
                    'id': 'we_01',
                    'title': 'Optimalkan Lingkungan Kerja Fisik',
                    'description': 'Perbaiki lingkungan kerja fisik termasuk pencahayaan, tingkat kebisingan, dan desain ruang kerja.',
                    'category': 'work_environment',
                    'priority': 'medium',
                    'implementation_steps': [
                        'Lakukan penilaian lingkungan tempat kerja',
                        'Perbaiki pencahayaan dan kurangi polusi suara',
                        'Buat zona kerja kolaboratif dan tenang',
                        'Tambahkan tanaman dan elemen alami ke ruang kerja'
                    ]
                },
                {
                    'id': 'we_02',
                    'title': 'Optimalisasi Teknologi dan Alat',
                    'description': 'Berikan teknologi dan alat yang memadai untuk membantu karyawan bekerja lebih efisien dan mengurangi frustrasi.',
                    'category': 'work_environment',
                    'priority': 'medium',
                    'implementation_steps': [
                        'Evaluasi teknologi saat ini dan identifikasi kesenjangan',
                        'Berikan pelatihan tentang alat dan sistem baru',
                        'Pastikan dukungan dan pemeliharaan IT yang handal',
                        'Terapkan solusi perangkat lunak yang ramah pengguna'
                    ]
                }
            ]
        }
        
        # Threshold definitions for dynamic selection
        self.thresholds = {
            'stress_level': {
                'low': 30,
                'medium': 60,
                'high': 70
            },
            'factors': {
                'workload': 50,
                'work_life_balance': 40,
                'team_conflict': 40,
                'management_support': 40,
                'work_environment': 45
            }
        }
    
    def get_dynamic_recommendations(self, 
                                  overall_stress: float,
                                  factors: Dict[str, float],
                                  department: str = None,
                                  num_recommendations: int = 3) -> List[Dict[str, Any]]:
        """
        Generate dynamic recommendations based on stress analysis
        
        Args:
            overall_stress: Overall stress level (0-100)
            factors: Dictionary of stress factors with their values
            department: Department name for context-specific recommendations
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of selected recommendations with implementation details
        """
        
        selected_recommendations = []
        priority_categories = self._determine_priority_categories(overall_stress, factors)
        
        # Get recommendations for each priority category
        for category, priority_level in priority_categories.items():
            category_recommendations = self._get_recommendations_by_category(category)
            
            # Filter by priority if high stress
            if overall_stress > self.thresholds['stress_level']['high']:
                category_recommendations = [r for r in category_recommendations 
                                          if r['priority'] == 'high']
            
            selected_recommendations.extend(category_recommendations)
        
        # If we don't have enough recommendations, add general stress management
        if len(selected_recommendations) < num_recommendations:
            general_recommendations = self.recommendations['stress_management']
            selected_recommendations.extend(general_recommendations)
        
        # Remove duplicates and shuffle for variety
        unique_recommendations = self._remove_duplicates(selected_recommendations)
        random.shuffle(unique_recommendations)
        
        # Select the required number of recommendations
        final_recommendations = unique_recommendations[:num_recommendations]
        
        # Add dynamic implementation steps based on context
        for rec in final_recommendations:
            rec = self._customize_recommendation(rec, overall_stress, factors, department)
        
        return final_recommendations
    
    def _determine_priority_categories(self, overall_stress: float, factors: Dict[str, float]) -> Dict[str, str]:
        """Determine which categories should be prioritized based on analysis"""
        priority_categories = {}
        
        # Check each factor against thresholds
        for factor_name, factor_value in factors.items():
            threshold = self.thresholds['factors'].get(factor_name, 50)
            
            if factor_value > threshold:
                if factor_name == 'workload':
                    priority_categories['workload_management'] = 'high'
                elif factor_name == 'work_life_balance':
                    priority_categories['work_life_balance'] = 'high' 
                elif factor_name == 'team_conflict':
                    priority_categories['team_communication'] = 'high'
                elif factor_name == 'management_support':
                    priority_categories['management_support'] = 'high'
                elif factor_name == 'work_environment':
                    priority_categories['work_environment'] = 'high'
        
        # Always include general stress management if overall stress is high
        if overall_stress > self.thresholds['stress_level']['high']:
            priority_categories['stress_management'] = 'high'
        elif overall_stress > self.thresholds['stress_level']['medium']:
            priority_categories['stress_management'] = 'medium'
            
        return priority_categories
    
    def _get_recommendations_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all recommendations for a specific category"""
        return self.recommendations.get(category, [])
    
    def _remove_duplicates(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate recommendations based on ID"""
        seen_ids = set()
        unique_recommendations = []
        
        for rec in recommendations:
            if rec['id'] not in seen_ids:
                seen_ids.add(rec['id'])
                unique_recommendations.append(rec)
                
        return unique_recommendations
    
    def _customize_recommendation(self, recommendation: Dict[str, Any], 
                                overall_stress: float, 
                                factors: Dict[str, float],
                                department: str = None) -> Dict[str, Any]:
        """Customize recommendation based on specific context"""
        
        # Add urgency level based on stress
        if overall_stress > self.thresholds['stress_level']['high']:
            recommendation['urgency'] = 'immediate'
        elif overall_stress > self.thresholds['stress_level']['medium']:
            recommendation['urgency'] = 'soon'
        else:
            recommendation['urgency'] = 'planned'
        
        # Add department-specific context if available
        if department:
            recommendation['department_context'] = self._get_department_context(
                recommendation['category'], department)
        
        # Add confidence score based on factor alignment
        recommendation['confidence_score'] = self._calculate_confidence_score(
            recommendation, factors)
        
        return recommendation
    
    def _get_department_context(self, category: str, department: str) -> str:
        """Get department-specific context for recommendations"""
        context_map = {
            ('workload', 'IT'): 'Pertimbangkan implementasi metodologi agile dan automated testing',
            ('workload', 'HR'): 'Fokus pada otomatisasi proses dan alat self-service karyawan', 
            ('workload', 'Sales'): 'Implementasikan alat CRM dan sistem prioritas lead',
            ('team_conflict', 'IT'): 'Dorong code review dan pair programming untuk kolaborasi',
            ('team_conflict', 'Sales'): 'Implementasikan insentif berbasis tim daripada kompetisi individual'
        }
        
        return context_map.get((category, department), 
                              f'Sesuaikan rekomendasi ini dengan kebutuhan departemen {department}')
    
    def _calculate_confidence_score(self, recommendation: Dict[str, Any], 
                                   factors: Dict[str, float]) -> float:
        """Calculate confidence score for recommendation relevance"""
        category = recommendation['category']
        
        # Map categories to factor names
        category_factor_map = {
            'workload': 'workload',
            'work_life_balance': 'work_life_balance', 
            'team_conflict': 'team_conflict',
            'management_support': 'management_support',
            'work_environment': 'work_environment',
            'general': 'overall'
        }
        
        related_factor = category_factor_map.get(category, 'overall')
        
        if related_factor in factors:
            factor_value = factors[related_factor]
            threshold = self.thresholds['factors'].get(related_factor, 50)
            
            # Higher confidence if factor value exceeds threshold significantly
            if factor_value > threshold:
                confidence = min(0.9, 0.6 + (factor_value - threshold) / 100)
            else:
                confidence = max(0.3, 0.6 - (threshold - factor_value) / 100)
        else:
            confidence = 0.5  # Default confidence
            
        return round(confidence, 2)


def get_recommendations_for_analysis(analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Main function to get recommendations based on analysis data
    
    Args:
        analysis_data: Analysis results containing stress levels and factors
        
    Returns:
        List of 3 dynamic recommendations
    """
    
    engine = DynamicRecommendationEngine()
    
    # Extract data from analysis
    overall_stress = analysis_data.get('overall_stress_level', 0)
    
    # Extract factor importance and convert to stress contribution values
    factor_importance = analysis_data.get('factor_importance', {})
    factors = {}
    
    for factor_name, factor_data in factor_importance.items():
        if isinstance(factor_data, dict):
            importance = factor_data.get('importance_percentage', 0)
            correlation = factor_data.get('correlation_with_stress', 0)
            
            # Convert importance to stress contribution (0-100 scale)
            stress_contribution = min(100, importance * abs(correlation) * 2)
            factors[factor_name] = stress_contribution
        else:
            # Handle simple numeric values
            factors[factor_name] = min(100, float(factor_data) if factor_data else 0)
    
    # Get department if available
    department = analysis_data.get('department', None)
    
    # Generate recommendations
    recommendations = engine.get_dynamic_recommendations(
        overall_stress=overall_stress,
        factors=factors,
        department=department,
        num_recommendations=3
    )
    
    return recommendations


# Test function for development
if __name__ == "__main__":
    # Example usage
    sample_analysis = {
        'overall_stress_level': 75,
        'factor_importance': {
            'workload': {'importance_percentage': 60, 'correlation_with_stress': 0.8},
            'work_life_balance': {'importance_percentage': 45, 'correlation_with_stress': -0.6},
            'team_conflict': {'importance_percentage': 30, 'correlation_with_stress': 0.4},
            'management_support': {'importance_percentage': 35, 'correlation_with_stress': -0.5},
        },
        'department': 'IT'
    }
    
    recommendations = get_recommendations_for_analysis(sample_analysis)
    
    print("Dynamic Recommendations Generated:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   Description: {rec['description']}")
        print(f"   Priority: {rec['priority']} | Urgency: {rec.get('urgency', 'N/A')}")
        print(f"   Confidence: {rec.get('confidence_score', 'N/A')}")
        print("   Implementation Steps:")
        for step in rec.get('implementation_steps', []):
            print(f"     - {step}") 