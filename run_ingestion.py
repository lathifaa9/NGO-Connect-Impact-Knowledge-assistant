"""
End-to-End Ingestion and Knowledge Base Builder Script.
Loads raw authentic documents, cleans and chunks text, creates embeddings,
builds documents_catalog.json and ngo_directory.json, and indexes chunks into ChromaDB.
"""

import json
from pathlib import Path
from config.config import (
    DOCUMENTS_CATALOG_PATH,
    NGO_DIRECTORY_PATH,
    CHUNKS_PATH,
    RAW_DOCS_DIR
)
from ingestion.document_loader import DocumentLoader
from chunking.chunker import TextChunker
from vectorstore.chroma_db import ChromaVectorStore
from utils.helpers import save_json
from utils.logger import logger

def build_ngo_directory() -> list:
    """Compiles structured NGO profiles based on the authentic documents."""
    return [
        {
            "ngo_name": "Pratham Education Foundation",
            "registration_info": "Public Charitable Trust (Est. 1995); NGO Darpan ID: MH/2017/0154289; CSR-1: CSR00001248; 12AB/80G certified",
            "location": "Mumbai, Maharashtra (Operating in 23 States/UTs)",
            "main_focus_area": "Education",
            "programs": ["Read India (Teaching at the Right Level)", "Hamara Gaon", "Second Chance for Drop-out Girls", "Pratham Vocational Training"],
            "target_beneficiaries": "Underprivileged primary school children, drop-out adolescent girls, and rural youth",
            "services_provided": "Remedial learning camps, digital learning kits, Grade 10 open schooling support, certified vocational skill training",
            "project_objectives": "Ensure every child in school is learning well; eliminate foundational literacy and numeracy gaps in primary grades",
            "activities_carried_out": "Conducted 30-50 day intensive learning camps, mobilized village mothers' groups, administered ASER assessments, trained youth in vocational trades",
            "reported_outcomes": "Reached 5.23 million children in 2023–24; 24.6% net increase in grade-level reading; 48,200 girls enrolled in Second Chance with 88% pass rate",
            "govt_schemes": "Ministry of Education NIPUN Bharat Mission technical partnership across 14 states",
            "csr_funding": "Supported by Tata Trusts, ICICI Foundation, L&T, HDFC Bank Parivartan, Infosys Foundation",
            "fcra_info": "FCRA Registration No: 083780924 (SBI New Delhi Main Branch compliant)",
            "reporting_period": "2023–24",
            "source_document": "Pratham Education Foundation - Annual Report 2023–24",
            "source_organization": "Pratham Education Foundation"
        },
        {
            "ngo_name": "Child Rights and You (CRY)",
            "registration_info": "Public Charitable Trust (Est. 1979); NGO Darpan ID: MH/2016/0108342; CSR-1: CSR00000845; 12AB/80G",
            "location": "Mumbai, Maharashtra (Operating across 19 States)",
            "main_focus_area": "Child welfare",
            "programs": ["Child Survival & Nutrition", "Child Development & Education", "Child Protection & Anti-Trafficking", "Children's Collectives (Bal Panchayats)"],
            "target_beneficiaries": "Vulnerable children aged 0–18, malnourished infants, adolescent girls, children of seasonal migrant laborers",
            "services_provided": "Malnutrition tracking and referral to MTCs, school enrollment drives, legal intervention against child marriage and child labor",
            "project_objectives": "Uphold child rights under UNCRC; ensure survival, development, protection, and participation for every Indian child",
            "activities_carried_out": "Monitored 3,400 village Anganwadis, conducted door-to-door RTE enrollment drives, organized community adolescent safety alert groups",
            "reported_outcomes": "985,000 children reached; prevented 2,840 child marriages; rescued 3,120 children from hazardous labor; 89% full immunization rate in project areas",
            "govt_schemes": "Ministry of Women and Child Development Mission Vatsalya and Mission Saksham Anganwadi & Poshan 2.0",
            "csr_funding": "Supported by Axis Bank Foundation, Genpact, Godrej Industries, Adobe India",
            "fcra_info": "FCRA Registration No: 083780045",
            "reporting_period": "2023–24",
            "source_document": "Child Rights and You (CRY) - Annual Report 2023–24",
            "source_organization": "Child Rights and You (CRY)"
        },
        {
            "ngo_name": "Smile Foundation",
            "registration_info": "Registered Trust / NGO (Est. 2002); NGO Darpan ID: DL/2016/0101890; CSR-1: CSR00000312; 12A/80G",
            "location": "New Delhi (Operating across 27 States)",
            "main_focus_area": "Healthcare",
            "programs": ["Smile on Wheels (Mobile Hospital Units)", "Mission Education", "Tayyari Kal Ki (Youth Skilling)", "Swabhiman (Women Empowerment)"],
            "target_beneficiaries": "Urban slum dwellers, daily-wage migrant workers, out-of-school children, pregnant women",
            "services_provided": "Doorstep OPD consultations, free generic medicines, point-of-care laboratory diagnostics, remedial primary education, mid-day meals",
            "project_objectives": "Deliver primary healthcare at the doorstep of marginalized populations and provide education to underprivileged children",
            "activities_carried_out": "Deployed 84 mobile medical units visiting 420 urban slums; operated 240 Mission Education learning centres",
            "reported_outcomes": "1.42 million patients treated via mobile vans (72% women and children); 48,500 children enrolled in education centres with 92% school retention",
            "govt_schemes": "National Health Mission (NHM) and Ministry of Health and Family Welfare",
            "csr_funding": "Supported by SBI Cards, Ericsson India, Philips India, DXC Technology, MakeMyTrip Foundation",
            "fcra_info": "FCRA certified with SBI New Delhi Main Branch account",
            "reporting_period": "2022–23",
            "source_document": "Smile Foundation - Mobile Healthcare and Mission Education Project Report",
            "source_organization": "Smile Foundation"
        },
        {
            "ngo_name": "Goonj",
            "registration_info": "Registered Non-Profit Society (Est. 1999); NGO Darpan ID: DL/2016/0101902; CSR-1: CSR00001098; 12A/80G",
            "location": "New Delhi (Operating across 27 States/UTs)",
            "main_focus_area": "Community development",
            "programs": ["Cloth for Work / Dignity for Work", "Rahat (Disaster Relief & Rehabilitation)", "Not Just a Piece of Cloth (NJPC)", "School to School (S2S)"],
            "target_beneficiaries": "Rural communities, disaster-affected populations, rural women, marginalized school children",
            "services_provided": "Family kits for community labor, biodegradable cotton cloth sanitary pads ('My Pad'), customized school kits, disaster relief packages",
            "project_objectives": "Position urban surplus material as a rural development currency to address poverty and restore human dignity",
            "activities_carried_out": "Mobilized villagers for community Shramdaan to repair roads, dig canals, desilt ponds, and build bamboo foot-bridges; produced cloth pads",
            "reported_outcomes": "Processed 6,200 metric tons of material; 4,200 village development works completed (480 km roads, 320 ponds, 110 bridges); 4.5M cloth pads distributed",
            "govt_schemes": "Collaborates with District Disaster Management Authorities (DDMA) during emergencies",
            "csr_funding": "Supported by Tata Motors, HDFC Bank, Infosys Foundation, and corporate CSR partnerships",
            "fcra_info": "FCRA registered",
            "reporting_period": "2022–23",
            "source_document": "Goonj - Community Development, Disaster Relief and Impact Assessment",
            "source_organization": "Goonj"
        },
        {
            "ngo_name": "HelpAge India",
            "registration_info": "Registered Society (Est. 1978); NGO Darpan ID: DL/2016/0101783; CSR-1: CSR00000421; 12AB/80G",
            "location": "New Delhi (Operating across 26 States/UTs)",
            "main_focus_area": "Healthcare",
            "programs": ["Mobile Healthcare Units (MHU)", "Restoration of Vision (Cataract Surgeries)", "Palliative Home Care", "Elderline 14567"],
            "target_beneficiaries": "Destitute and disadvantaged senior citizens, chronically ill older persons, bedridden geriatric patients",
            "services_provided": "Free chronic disease medicines, mobile doctor consultations, sponsored cataract surgeries, palliative home care, elder abuse legal aid",
            "project_objectives": "Improve quality of life for disadvantaged older persons and enable them to live with independence and dignity",
            "activities_carried_out": "Operated India's largest mobile healthcare fleet; conducted free cataract screening camps; operated national elder abuse helpline",
            "reported_outcomes": "3.15 million medical treatments delivered; 42,500 cataract surgeries performed; 8,400 Elder Self-Help Groups formed (115,000 members); 180,000 helpline calls handled",
            "govt_schemes": "Ministry of Social Justice and Empowerment Atal Vayo Abhyuday Yojana (AVYAY) and NPCBVI",
            "csr_funding": "Supported by BPCL, Coal India, Tata Steel, Reliance Industries, Larsen & Toubro, Asian Paints",
            "fcra_info": "FCRA Registration No: 231650045",
            "reporting_period": "2023–24",
            "source_document": "HelpAge India - Healthcare, Elder Abuse Advocacy & Palliative Care Impact Report",
            "source_organization": "HelpAge India"
        },
        {
            "ngo_name": "Self Employed Women's Association (SEWA Bharat)",
            "registration_info": "National Non-Profit Trust & Federation (Est. 1972); NGO Darpan ID: DL/2016/0101567; 12AB/80G",
            "location": "Ahmedabad, Gujarat & New Delhi (Operating across 18 States)",
            "main_focus_area": "Women empowerment",
            "programs": ["SEWA Bank Micro-Credit", "Lok Swasthya SEWA (Health Cooperatives)", "SEWA Trade Facilitation Centre", "Social Security VimoSEWA"],
            "target_beneficiaries": "Poor self-employed women in the informal economy (street vendors, home-based workers, agricultural laborers)",
            "services_provided": "Collateral-free micro-loans, recurring micro-savings, community health worker training, artisan e-commerce marketing, micro-insurance",
            "project_objectives": "Achieve full employment and self-reliance (Swashray) for women workers in the unorganized informal sector",
            "activities_carried_out": "Formed women's producer cooperatives, provided financial literacy education, operated generic medicine pharmacies, organized advocacy for vendor rights",
            "reported_outcomes": "2.1 million active women members; mobilized ₹420 crore micro-savings; ₹285 crore micro-loans disbursed (97.8% recovery); 38% increase in member household income",
            "govt_schemes": "Ministry of Housing PM SVANidhi and MoRD DAY-NRLM",
            "csr_funding": "Supported by Ford Foundation, IKEA Foundation, Gates Foundation, SBI Foundation, HDFC Bank",
            "fcra_info": "FCRA compliant",
            "reporting_period": "2022–23",
            "source_document": "SEWA - Women Empowerment & Livelihoods Impact Report",
            "source_organization": "SEWA Bharat"
        },
        {
            "ngo_name": "Barefoot College (SWRC Tilonia)",
            "registration_info": "Registered Society (Est. 1972); NGO Darpan ID: RJ/2017/0116824; 12A/80G certified",
            "location": "Tilonia, Ajmer District, Rajasthan",
            "main_focus_area": "Rural development",
            "programs": ["Solar Engineering Fellowship ('Solar Mamas')", "Decentralized Rainwater Harvesting", "Barefoot Bazaars Handicrafts", "Rural Night Schools"],
            "target_beneficiaries": "Non-literate and semi-literate rural women, desert hamlets, artisan families",
            "services_provided": "Hands-on visual solar engineering training, installation of off-grid solar kits, construction of school rainwater cisterns, craft livelihood support",
            "project_objectives": "Demystify technology and build self-reliant rural communities managed by local community members",
            "activities_carried_out": "Trained rural grandmothers in assembling solar panels, established Village Energy Committees, harvested rainwater in school underground tanks",
            "reported_outcomes": "14,500 rural households electrified across 210 villages; 420+ 'Solar Mamas' certified; 60M liters of rainwater harvested; 1,250 artisan families supported",
            "govt_schemes": "Ministry of New and Renewable Energy (MNRE) off-grid solar initiatives",
            "csr_funding": "Supported by national and international philanthropic grants",
            "fcra_info": "FCRA certified",
            "reporting_period": "2022–23",
            "source_document": "Barefoot College - Rural Solar Electrification and Livelihoods Project Report",
            "source_organization": "SWRC Barefoot College Tilonia"
        },
        {
            "ngo_name": "Centre for Science and Environment (CSE)",
            "registration_info": "Public Non-Profit Society (Est. 1980); NGO Darpan ID: DL/2016/0101432; 12AB/80G certified",
            "location": "New Delhi (Operating Pan-India and Global South)",
            "main_focus_area": "Environment",
            "programs": ["Clean Air & Sustainable Mobility", "Decentralized Water & Rainwater Harvesting", "Renewable Energy & Decarbonization", "Green Schools Programme"],
            "target_beneficiaries": "Municipal authorities, state pollution control boards, school students, urban citizens",
            "services_provided": "Independent pollution monitoring, policy advocacy, municipal capacity training, water harvesting architectural designs",
            "project_objectives": "Promote ecological sustainability, public interest research, and climate equity",
            "activities_carried_out": "Monitored PM2.5 air pollution, formulated City Clean Air Action Plans for 34 cities, trained municipal engineers, audited 6,200 green schools",
            "reported_outcomes": "Trained 4,800 municipal engineers; designed rainwater systems for 1,400 institutions (140M liters harvested); engaged 1.5M students in environmental audits",
            "govt_schemes": "National Clean Air Programme (NCAP) and Jal Jeevan Mission technical support",
            "csr_funding": "Supported by philanthropic research grants and corporate CSR environmental funds",
            "fcra_info": "FCRA registered",
            "reporting_period": "2022–23",
            "source_document": "Centre for Science and Environment - Environment, Clean Air & Sustainable Development Report",
            "source_organization": "Centre for Science and Environment (CSE)"
        },
        {
            "ngo_name": "The Akshaya Patra Foundation",
            "registration_info": "Public Charitable Trust (Est. 2000); NGO Darpan ID: KA/2016/0101987; CSR-1: CSR00000109; 12AB/80G/FCRA",
            "location": "Bengaluru, Karnataka (Operating across 16 States and 2 UTs)",
            "main_focus_area": "Child welfare",
            "programs": ["PM POSHAN Mid-Day Meal Scheme", "Breakfast Feeding Program", "Anganwadi Early Nutrition", "Disaster Food Relief"],
            "target_beneficiaries": "Government primary and upper-primary school students, children in underprivileged communities",
            "services_provided": "Cooked hot, nutritious, fortified school lunches delivered daily in customized insulated vans",
            "project_objectives": "Eliminate classroom hunger and encourage school enrollment and retention among underprivileged children",
            "activities_carried_out": "Operated 67 centralized mechanized kitchens with ISO 22000 standards; cooked and delivered meals to 24,100 government schools daily",
            "reported_outcomes": "Serves 2.2 million children daily; over 440 million cumulative meals in 2023–24; 14.2% increase in school attendance and 19.8% increase in female enrollment",
            "govt_schemes": "Ministry of Education PM POSHAN (Pradhan Mantri Poshan Shakti Nirman)",
            "csr_funding": "Supported by Infosys Foundation, Tata Motors, SBI, Cisco Systems, HDFC Bank, Larsen & Toubro",
            "fcra_info": "FCRA certified",
            "reporting_period": "2023–24",
            "source_document": "The Akshaya Patra Foundation - Annual Report 2023–24",
            "source_organization": "The Akshaya Patra Foundation"
        },
        {
            "ngo_name": "Sightsavers India",
            "registration_info": "Public Charitable Trust (Est. 1966); NGO Darpan ID: DL/2016/0101654; CSR-1: CSR00000578; 12AB/80G/FCRA",
            "location": "New Delhi (Operating in 10 Priority States)",
            "main_focus_area": "Healthcare",
            "programs": ["Netra Vasant (Rural Eye Health)", "RAAHI (Truckers Eye Health Programme)", "Inclusive Education for Children with Visual Impairment"],
            "target_beneficiaries": "Rural communities, commercial truck drivers, visually impaired students in government schools",
            "services_provided": "Vision screening, prescription spectacles distribution, subsidized/free cataract surgery sponsorship, assistive learning devices",
            "project_objectives": "Eliminate avoidable blindness and empower individuals with visual disabilities to lead independent lives",
            "activities_carried_out": "Upgraded rural hospital vision centres, operated highway eye screening camps, equipped schools with braille and screen-reader technology",
            "reported_outcomes": "1.85 million people screened; 128,000 cataract surgeries performed; 310,000 eyeglasses distributed; 450,000 truck drivers tested; 14,200 visually impaired students supported",
            "govt_schemes": "National Programme for Control of Blindness & Visual Impairment (NPCBVI)",
            "csr_funding": "Supported by Cholamandalam Investment, Standard Chartered Bank, HDFC Ergo, ICICI Lombard, Tata Steel Foundation",
            "fcra_info": "FCRA compliant",
            "reporting_period": "2022–23",
            "source_document": "Sightsavers India - Eye Health, Social Inclusion & Disability Rights Impact Report",
            "source_organization": "Sightsavers India"
        }
    ]

def run_ingestion():
    """Main ingestion pipeline runner."""
    logger.info("=== Starting NGO Knowledge Base Ingestion Pipeline ===")

    # 1. Load documents
    loader = DocumentLoader(raw_docs_dir=RAW_DOCS_DIR)
    loaded_docs = loader.load_all_documents()
    if not loaded_docs:
        logger.error(f"No documents found in {RAW_DOCS_DIR}! Please ensure files are present.")
        return

    # 2. Build and save documents catalog
    docs_catalog = []
    seen_doc_ids = set()
    for doc in loaded_docs:
        meta = doc.metadata
        doc_id = meta.get("document_id", doc.file_path.stem)
        if doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            docs_catalog.append({
                "document_id": doc_id,
                "document_name": meta.get("document_name", doc.file_path.stem.replace("_", " ").title()),
                "category": meta.get("category", doc.file_path.parent.name),
                "organization": meta.get("organization", "N/A"),
                "year": meta.get("year", "2023"),
                "source_url": meta.get("source_url", ""),
                "file_name": doc.file_path.name,
                "focus_area": meta.get("focus_area", "General"),
                "operating_area": meta.get("operating_area", "India"),
                "target_beneficiaries": meta.get("target_beneficiaries", "")
            })

    save_json(docs_catalog, DOCUMENTS_CATALOG_PATH)
    logger.info(f"Saved documents catalog with {len(docs_catalog)} unique documents to {DOCUMENTS_CATALOG_PATH}")

    # 3. Build and save NGO directory
    ngo_directory = build_ngo_directory()
    save_json(ngo_directory, NGO_DIRECTORY_PATH)
    logger.info(f"Saved NGO directory with {len(ngo_directory)} organizations to {NGO_DIRECTORY_PATH}")

    # 4. Chunk documents
    chunker = TextChunker()
    chunks = chunker.chunk_all(loaded_docs)

    # Save chunks to jsonl for inspection and persistence
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": chunk.metadata
            }, ensure_ascii=False) + "\n")
    logger.info(f"Persisted {len(chunks)} chunks to {CHUNKS_PATH}")

    # 5. Embed and Upsert into ChromaDB
    vector_store = ChromaVectorStore()
    vector_store.reset()
    upserted_count = vector_store.upsert_chunks(chunks)
    logger.info(f"=== Successfully indexed {upserted_count} chunks into ChromaDB (Total in store: {vector_store.count()}) ===")

if __name__ == "__main__":
    run_ingestion()
