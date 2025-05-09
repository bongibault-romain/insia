import csv
import threading
import time
import tkinter as tk
from tkinter import messagebox

def generate_text(question):
    time.sleep(2)  # Simule un long traitement
    return f"Réponse générée pour: {question}"

# Questions à traiter
questions = """
Quels sont les documents nécessaires pour finaliser mon inscription administrative ?
Comment obtenir une attestation de scolarité ?
Où puis-je récupérer ma carte étudiante ?
Comment modifier mes informations personnelles sur le portail étudiant ?
Quel est le calendrier universitaire de cette année ?
Comment faire une demande de bourse CROUS ?
Quels sont les critères d’attribution des logements en résidence étudiante ?
Puis-je changer de chambre en cours d’année ?
Où puis-je consulter mes résultats d’examen ?
Comment contester une note ?
Quelles sont les modalités de rattrapage ?
Quelles sont les conditions pour valider mon année ?
Est-ce possible de changer de groupe de TP ?
Comment faire appel d’une décision du jury ?
Comment obtenir une convention de stage ?
Quels sont les délais pour faire valider un stage ?
Y a-t-il des offres de stage disponibles sur le campus ?
Où trouver le bureau des stages ?
Puis-je faire mon stage à l’étranger ?
Quelles sont les démarches pour une mobilité internationale ?
Quelles sont les universités partenaires de l’INSA Toulouse ?
Comment candidater à un échange Erasmus ?
Quels sont les critères de sélection pour un double diplôme ?
Combien coûte une année à l’étranger ?
Puis-je bénéficier d’aides financières pour partir en échange ?
Comment obtenir mon relevé de notes officiel ?
Quand ont lieu les soutenances de projet ?
Quels sont les délais pour remettre mon mémoire de fin d’études ?
Où trouver les sujets de PFE des années précédentes ?
Quelles sont les règles pour les absences aux cours obligatoires ?
Comment déclarer une absence justifiée ?
Comment obtenir un certificat médical pour l’administration ?
Quelle est la procédure en cas de redoublement ?
Comment se réinscrire si je redouble ?
Quelles sont les dates d’ouverture des candidatures pour les masters ?
Comment candidater à un master INSA après le cycle ingénieur ?
Quels sont les horaires du service de scolarité ?
Où puis-je trouver un emploi étudiant sur le campus ?
Y a-t-il des aides pour les étudiants en difficulté financière ?
Comment obtenir un rendez-vous avec une assistante sociale ?
Comment fonctionne le tutorat entre étudiants ?
Puis-je changer de filière en cours de cycle ingénieur ?
Comment s’inscrire aux activités sportives proposées par l’INSA ?
Quels sports sont disponibles sur le campus ?
Y a-t-il des compétitions sportives inter-écoles ?
Comment rejoindre une association étudiante ?
Où consulter la liste des associations étudiantes ?
Puis-je créer une nouvelle association ?
À qui dois-je m’adresser pour organiser un événement sur le campus ?
Comment réserver une salle pour une réunion d’association ?
Quelles sont les démarches pour obtenir un visa étudiant ?
Quelles sont les aides pour les étudiants internationaux ?
Y a-t-il des cours de français pour les étudiants étrangers ?
Comment fonctionne la plateforme Moodle de l’INSA ?
Que faire si j’ai perdu mon mot de passe ENT ?
Où trouver mon emploi du temps ?
Comment sont organisés les examens ?
Quelles sont les règles pendant les examens ?
Puis-je passer un examen en décalé ?
Comment demander un aménagement d’examen ?
Quelles sont les procédures en cas de fraude aux examens ?
Quels sont les critères pour l’obtention du diplôme d’ingénieur ?
Quand a lieu la remise des diplômes ?
Puis-je obtenir un duplicata de diplôme ?
Comment m’inscrire à la cérémonie de remise des diplômes ?
Que faire si je perds ma carte étudiante ?
Comment accéder aux bâtiments en dehors des horaires classiques ?
Quels sont les horaires du restaurant universitaire ?
Comment recharger ma carte IZLY ?
Comment signaler un problème dans ma chambre de résidence ?
Quelles sont les consignes en cas d’alarme incendie ?
Quelles aides sont disponibles pour un étudiant en situation de handicap ?
Comment demander un logement prioritaire pour raison de santé ?
Y a-t-il une cellule de soutien psychologique pour les étudiants ?
Où puis-je trouver les règlements intérieurs des résidences ?
Qu’est-ce que le tronc commun à l’INSA ?
Comment choisir ma spécialité d’ingénieur ?
Quels sont les débouchés pour chaque spécialité ?
Y a-t-il des conférences métiers organisées par l’école ?
Comment m’inscrire à une conférence ou à un atelier professionnel ?
Puis-je obtenir des crédits ECTS pour une activité associative ?
Quelles sont les formations complémentaires proposées (MOOC, certifications) ?
Comment demander une année de césure ?
Quels sont les avantages d’une année de césure ?
Comment préparer mon projet de césure ?
Puis-je cumuler une activité salariée avec mes études ?
Comment demander une exonération des frais de scolarité ?
Comment déclarer un changement de situation familiale ?
Puis-je faire une alternance à l’INSA Toulouse ?
Quels sont les critères pour intégrer un parcours en alternance ?
Comment sont évalués les projets de groupe ?
Comment accéder à la bibliothèque ?
Puis-je emprunter un ordinateur portable ?
Quels logiciels sont disponibles gratuitement pour les étudiants ?
Comment faire un signalement en cas de harcèlement ?
Où puis-je consulter les offres d’emploi post-diplôme ?
Comment rejoindre le réseau des anciens élèves ?
Quels sont les dispositifs d’égalité et d’inclusion à l’INSA ?
Y a-t-il un service d’aide à la rédaction de CV/lettres de motivation ?
Comment s’inscrire à une simulation d’entretien ?
"""
questions = list(map(lambda x: x.strip(), questions.strip().split("\n")))

# Dictionnaire pour stocker les réponses générées
generated_answers = {}
answers_lock = threading.Lock()

# Thread de génération en arrière-plan
def background_generation():
    for question in questions:
        with answers_lock:
            if question not in generated_answers:
                generated_answers[question] = None  # Marquer comme en cours
        answer = generate_text(question)
        with answers_lock:
            generated_answers[question] = answer

class EvaluationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Évaluation de Réponses")
        self.index = 0
        self.scores = []
        self.answer_available = False

        self.master.bind("<Return>", lambda event: self.save_score_and_next())

        self.question_label = tk.Label(master, text="", wraplength=500, font=("Arial", 12, "bold"))
        self.question_label.pack(pady=10)

        self.answer_label = tk.Label(master, text="", wraplength=500, font=("Arial", 11), fg="gray")
        self.answer_label.pack(pady=10)

        self.score_entry = tk.Entry(master)
        self.score_entry.pack(pady=10)

        self.next_button = tk.Button(master, text="Suivant", command=self.save_score_and_next)
        self.next_button.pack(pady=10)

        self.display_question()

    def display_question(self):
        if self.index >= len(questions):
            self.export_results()
            messagebox.showinfo("Fini", "Toutes les questions ont été évaluées.")
            self.master.quit()
            return

        self.question = questions[self.index]
        self.question_label.config(text=f"Question : {self.question}")
        self.score_entry.delete(0, tk.END)
        self.answer_label.config(text="Chargement de la réponse...", fg="gray")
        self.answer_available = False
        self.next_button.config(state="disabled")

        # Vérifie périodiquement si la réponse est prête
        self.master.after(200, self.check_answer_ready)

    def check_answer_ready(self):
        with answers_lock:
            answer = generated_answers.get(self.question)

        if answer:
            self.answer_label.config(text=f"Réponse : {answer}", fg="black")
            self.next_button.config(state="normal")
            self.answer_available = True
        else:
            self.master.after(200, self.check_answer_ready)

    def save_score_and_next(self):
        score_str = self.score_entry.get()

        if not self.answer_available:
            messagebox.showerror("Erreur", "Veuillez attendre que la réponse charge.")
            return

        if not score_str.isdigit() or not (0 <= int(score_str) <= 100):
            messagebox.showerror("Erreur", "Veuillez entrer un score entre 0 et 100.")
            return

        self.answer_available = False
        self.scores.append(int(score_str))
        self.index += 1
        self.display_question()

    def export_results(self):
        with open("resultats.csv", mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for score in self.scores:
                writer.writerow([score])

if __name__ == "__main__":
    threading.Thread(target=background_generation, daemon=True).start()

    root = tk.Tk()
    app = EvaluationApp(root)
    root.mainloop()
