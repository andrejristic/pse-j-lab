# PSE J-Lab — vodič

Poređano po važnosti. Prvih pet stvari pročitaj pre puta; ostalo kad zatreba.

---

## 1. Prvo pokretanje MORA biti kod kuće

Guardian uči TLS otisak svakog servera pri prvom kontaktu. Ako ga naučiš na
hotelskom wifiju, naučiće ono što ta mreža pokaže — i posle će toj mreži
verovati. Zato:

```
cd ~/Desktop/PSE/PSE\ J-Lab/app
./setup.sh                 # gradi okruženje iz priloženih wheels-a, bez mreže
./run.sh --selftest        # 20 provera, ~5 s
./run.sh                   # prvi put: čarobnjak → dugme "Nauči TLS otiske sada"
```

Ako je `--selftest` pao bilo gde, **ne veruj nijednom broju koji aplikacija
ispiše**, dok ne vidiš šta je palo.

## 2. Šta ova aplikacija zapravo tvrdi

Da je Jakobijan mreže egzaktno izračunljiv bez autodiff biblioteke:

```
J(x) = W_L · D_{L-1}(x) · W_{L-1} · … · D_1(x) · W_1,   D_l = diag(φ'(z_l))
```

i da je sve ostalo posledica toga. Ta tvrdnja je proverena na četiri načina:
konačnim razlikama, poređenjem forward/reverse akumulacije, poznatim tačnim
odgovorima (Fuss–Catalan, log|det J| kod coupling tokova), i — ako instaliraš
torch — direktnim poređenjem sa `torch.func.jacrev`. Poslednje daje slaganje
na **1e-16**, uključujući ručno izveden gradijent Jakobijan-kazne.

Torch nije potreban ni za jedan lab. Služi samo kao nezavisni svedok:

```
./setup.sh --with-torch     # ~1 GB, offline iz wheels-a
./run.sh --selftest         # sad uključuje 30 poređenja sa torch-om
```

## 3. Pet laboratorija — i šta u svakoj može da pođe naopako

| Lab | Pitanje | Gde je zamka |
|---|---|---|
| **L1 Izometrija** | kako spektar J zavisi od dubine, inicijalizacije, aktivacije | χ=1 ne znači da mreža može da se trenira — kondicioni broj svejedno eksponencijalno raste |
| **L2 NTK** | spektar Θ=J_θJ_θᵀ, poravnanje sa labelama, pomeranje kernela | korak učenja **mora** biti ispod 2n/λ_max, inače porediš dve divergencije (zato je `lr:"auto"`) |
| **L3 Robusnost** | ‖J‖₂ i margina/gradijent naspram stvarnog PGD napada | robusnu tačnost meri na **više** jačina napada; na jednoj dobiješ 100% ili 0% i ništa ne vidiš |
| **L4 Dimenzija** | rang J autoenkodera naspram poznate dimenzije | procenitelj mora ignorisati numerički rep ispod ranga bottleneck-a, inače uvek javi širinu bottleneck-a |
| **L5 log\|det J\|** | analitički vs sklopljeni vs numerički determinant | ovo je najstroža provera engine-a; ako padne, nijedan drugi lab ne vredi |

Pokretanje iz GUI-ja: tab **Laboratorija** → izaberi levo → uredi parametre (JSON)
→ Pokreni. Ili iz terminala:

```
./run.sh --list
./run.sh --lab l1_depth_sweep --params '{"depths":[2,4,8,16,32],"width":64}'
```

## 4. Jedna kapija — zašto ti aplikacija odbija tvrdnje

Nijedna tvrdnja ne ulazi u trezor bez pokrića:

- **COMPUTED** mora pokazati na direktorijum sa rezultatom čiji se sha256 poklapa.
  Ako posle ručno izmeniš `result.json`, provera to otkrije i kartica pada.
- **CITED** mora imati referencu koja se **razreši preko mreže**. Naslov koji se
  ne poklapa znači da je ID pogrešno zapamćen — predlog se odbija, ne ispravlja.
- **SPECULATION** ne sme nositi dokaz. Ako ga ima, nije spekulacija.
- **REFUTED** se nikad ne briše.

Odbijanje nije kvar — to je aplikacija koja radi. Trezor koji ništa ne odbija
je sveska.

Model **nikada** ne upisuje karticu. On predlaže; ti upisuješ.

## 5. Guardian — jer si na tuđoj mreži

Sav saobraćaj ide kroz jednu funkciju. Sve ostalo je zabranjeno:
samo https, samo domeni sa liste, preusmerenja se proveravaju ručno na svakom
skoku, API ključ ide isključivo na endpoint svog provajdera, i svaki dokument
koji stigne skenira se na prompt-injection pre nego što ga model vidi.

Tab **Guardian** pokazuje stanje. Ako dugo ništa nije blokirano ni označeno,
aplikacija ti to kaže — Guardian koji ništa ne primećuje nije budan.

---

## 6. Šta je gde

```
PSE J-Lab/
  app/            kod, run.sh, setup.sh
  datasets/       MNIST + FashionMNIST + MANIFEST.json sa heševima
  wheels/         311 MB paketa za offline instalaciju
~/.local/share/pse-jlab/
  runs/           svaki eksperiment: params, result.json, sha256, meta, plots
  vault/          kartice (SQLite) + teme + predlozi referenci
  state/          journal.sqlite (append-only), theta.json, keys.json (0600)
```

## 7. Modeli (opciono)

Nijedan lab ne traži mrežu. Ključevi se unose u čarobnjaku ili u tabu Razgovor.
Ruter bira po ulozi, i **skeptika šalje kod drugog provajdera** nego autora —
model koji ocenjuje sopstveni rad se slaže sam sa sobom.

Cene: samo Anthropic ima stvarnu cenu po tokenu (opus-5 $5/$25 na milion).
Ostali se beleže kao $0 i to je **označeno**, jer izmišljen broj u izveštaju
izgleda kao znanje.

## 8. Izveštaj

Tab **Izvestaj** → Napravi → Sačuvaj. Dobiješ markdown u kojem svaka COMPUTED
tvrdnja nosi run id, sha256 i rezultat provere heša. To je ono što nosiš kući
s puta.

## 9. Ako nešto pukne

```
./run.sh --selftest              # prvo ovo, uvek
```

- **`no interpreter with numpy found`** → nisi pokrenuo `./setup.sh`.
- **`datasets present…` pada** → proveri `datasets/MANIFEST.json`; podaci su se
  promenili ili ih nema. Ne pokušavaj da ih ponovo skidaš sa hotelskog wifija.
- **theta van opsega posle nadogradnje** → aplikacija sama resetuje theta ako je
  snimljena pod drugim konstantama; u dnevniku stoji `THETA_RESET`.
- **Guardian javlja `TLS pin mismatch`** → ključ *i* izdavač sertifikata su se
  promenili. Na tuđoj mreži: ne zaobilazi ovo.
- **Ekran na 200% — sve iskače iz prozora** → prozori se sami prilagođavaju
  raspoloživom ekranu, a sadržaj čarobnjaka ima skrol sa dugmadima fiksiranim
  dole. Ako ipak hoćeš više sadržaja nego što skaliranje dozvoljava, pregazi ga
  **samo za ovu aplikaciju**:
  ```
  PSE_JLAB_SCALE=1     ./run.sh     # kao da je 100% (sitnije, ali sve staje)
  PSE_JLAB_SCALE=1.5   ./run.sh     # na pola puta
  ```
  Ovo ne dira sistemsko podešavanje.

## 10. Instalacija na drugu mašinu

```
sudo apt install ./pse-jlab_0.1.0_all.deb    # 345 MB, nosi i podatke i wheels
/opt/pse-jlab/app/setup.sh
pse-jlab --selftest
```
