# Math Solver — première passe de fiabilisation

29 septembre 2026. Analyse du dépôt local, reproductions ciblées et tests hors réseau. D’autres chats modifient simultanément les benchmarks, RRSI et l’interface : cette note distingue les corrections de cette passe des chantiers à poursuivre.

## Bugs reproduits et corrigés dans cette passe

| Zone | Défaut constaté | Correction |
| --- | --- | --- |
| Interface | Le `try` du bouton Reset n’était pas refermé : tout `app.js` échouait à l’analyse syntaxique. | Fermeture du bloc et traitement explicite des erreurs HTTP. |
| Vérification | `1+1 = 2` puis `\boxed{999}` suffisait à obtenir VERIFIED. | Les étapes locales seules ne certifient plus la réponse finale ; une référence est requise. Le statut indique sa base de vérification. |
| Lecture des maths | Fractions imbriquées ignorées, parenthèses LaTeX supprimées, exposants mal groupés, congruences fractionnaires tronquées en entiers. | Extraction équilibrée des accolades, conservation du groupement et exigence d’entiers exacts pour les congruences. |
| Évaluation des expressions | Des chaînes externes passaient directement dans `sympify`. | Grammaire arithmétique explicite via AST, sans évaluation de code Python, avec bornes de taille. Appliquée au vérificateur texte, aux expressions SymPy et aux propositions soumises à Z3. |
| Solveur exact | Le solveur récupérait une expression isolée dans un énoncé et la prenait pour sa réponse ; le cas polynomial ignorait le polynôme réellement demandé. | Reconnaissance de l’énoncé complet, refus des variantes non prises en charge et arithmétique sur l’expression complète. |
| Divisibilité | `not divisible by 4` était aussi ajouté comme contrainte positive. | Analyse séparée des exclusions, bornes `<` et `<=`, refus des diviseurs nuls et des contraintes non reconnues. |
| Cache | `('ab','c')` et `('a','bc')` avaient la même clé ; la casse était perdue. | Sérialisation de la liste d’arguments avant hachage, conservation des types et de la casse. |
| API | Les calculs synchrones étaient appelés sur la boucle asynchrone ; une résolution gelait les lectures. | Exécution des POST dans les workers FastAPI et verrou partagé pour les opérations modifiant l’état. GET reste disponible ; une opération concurrente reçoit 409. |
| Validation API | Moteurs inconnus, nombres de tokens négatifs et boucles sans borne acceptés. | Validation Pydantic des moteurs, tailles et intervalles. |
| Persistance | Plusieurs objets indépendants pouvaient réécrire le même graphe ; certains fichiers étaient enregistrés dans un ancien répertoire Antigravity. | Partage du harness API avec benchmark, boucle RRSI et crawler ; chemins centralisés dans DATA_DIR. |
| Graphe | SymPy/Z3 étaient présentés comme Violetto ; un résultat non vérifié pouvait être affiché SOUND par défaut. | Types et libellés corrects des moteurs ; statuts explicites ; affichage direct de la solution après résolution. |
| Exécution externe | Un échec ou une sortie vide de Codex pouvait devenir une réponse ordinaire ; aucune limite de temps. | Contrôle du code de sortie, rejet des sorties vides, délai maximal et utilisation de CODEX_BIN. |
| Affichage mathématique | KaTeX ignorait les preuves contenues dans les balises `pre`. | Activation du rendu mathématique dans ces blocs. |

## Surface d’amélioration, par ordre de priorité

1. **Contrat mathématique.** Séparer réponse exacte, identité locale, preuve complète et résultat empirique. Rendre explicites les hypothèses et domaines ; les équations avec variables peuvent être des affectations ou des conséquences d’hypothèses. Le vérificateur actuel n’est pas un assistant de preuve formelle. Le parcours naturel reste surtout anglophone ; des tests français sont nécessaires.
2. **Évaluation indépendante.** Les six cas historiques sont des tests de fonctionnement de quelques patrons reconnus, pas une mesure de performance olympique. Étendre les données, vérifier leur provenance, isoler apprentissage/réglage/test, ajouter des variantes adversariales et publier les résultats par moteur et par domaine. Chantier benchmark parallèle en cours.
3. **RRSI mesurable.** Le code initial testait presque exclusivement le raccourci SymPy : les mutations de prompt, de routage et de retrieval n’étaient pas exercées. Les mesures de latence portaient sur le dernier cas. Un autre chat traite les chemins d’évaluation, la stratégie de recherche et les comparaisons ; il reste à démontrer les gains sur des tests indépendants, plusieurs seeds et des budgets comparables.
4. **Exécution bornée.** Ajouter un budget global et une annulation effective couvrant routage, littérature, modèles et vérification. Le timeout Codex et les limites du parser ne remplacent pas l’isolation des calculs symboliques coûteux. Tester réellement les indisponibilités JEV, MPS et Codex et le respect des moteurs forcés.
5. **Persistance et reproductibilité.** Les JSON et verrous actuels ciblent un serveur local unique. Plusieurs processus, des fichiers corrompus ou un arrêt pendant une opération exigent des transactions et une récupération explicite. Les clés de cache de génération devraient inclure la version du modèle et du prompt. Ajouter un manifeste de dépendances et une commande d’installation reproductible.
6. **Littérature et interface.** Le client arXiv dépend de regex sur du HTML ; conserver les sources, dates et passages cités, puis vérifier leur rapport avec les conclusions. L’interface doit montrer une progression issue de l’exécution réelle, les limites des résultats et les erreurs de dépendances. Le navigateur signale encore un double chargement de Three.js. Chantier UX parallèle en cours.

Les anciens résultats persistés et générations RRSI n’ont pas été recalculés : leur présence dans le graphe n’est pas une validation rétroactive.

## Validation de cette passe

```sh
python3 check_reliability.py
python3 -m pytest -q tests/test_core.py
node --check web/app.js
python3 -m compileall -q .
```

Le contrôle hors réseau utilise un répertoire temporaire et couvre les sept calculs exacts (six exemples historiques et une addition de fractions), les faux positifs, les énoncés non reconnus, les entrées dangereuses, le cache, la provenance du graphe, la validation API et les lectures pendant une opération lente simulée.

Les 10 tests du chantier RRSI passent également, exécutés avec un DATA_DIR temporaire. Une dépréciation PyPDF2 provient des dépendances.

Après relance du serveur local, le parcours navigateur « Compute 1 / 2 + 1 / 3 » affiche `5/6`, attribué à `sympy_cas`. Aucun benchmark complet de qualité des modèles, essai MPS prolongé, appel Codex réel ou parcours arXiv complet n’est inclus dans cette validation.
